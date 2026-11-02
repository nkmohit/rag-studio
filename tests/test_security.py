import asyncio
import json
import re
from io import BytesIO

import pytest
from fastapi import HTTPException, UploadFile
from fastapi.testclient import TestClient
from pypdf import PdfWriter

import main
from utils.rag import embeddings
from utils.rag.embeddings import chunk_id, chunk_text

client = TestClient(main.app)

UNSAFE_NAMES = [
    "../secrets.txt",
    "..\\secrets.txt",
    "../../etc/passwd.txt",
    "/etc/passwd.txt",
    "foo/bar.txt",
    "foo..txt",
    "notes.exe",
    "notes.txt.exe",
    ".secrets.txt",
    "a/../notes.txt",
]


@pytest.mark.parametrize("endpoint", ["/upload", "/index"])
@pytest.mark.parametrize("name", UNSAFE_NAMES)
def test_rejects_unsafe_uploads(endpoint, name, isolated_storage):
    outside = isolated_storage.parent / "secrets.txt"
    response = client.post(
        endpoint,
        files={"file": (name, b"owned", "text/plain")},
    )
    assert response.status_code == 400
    assert not outside.exists()
    assert list(isolated_storage.iterdir()) == []


def test_rejects_empty_filename_and_parent_delete(isolated_storage):
    with pytest.raises(HTTPException) as empty_name:
        asyncio.run(main.upload_file(UploadFile(filename="", file=BytesIO(b"hello"))))
    assert empty_name.value.status_code == 400

    with pytest.raises(HTTPException) as parent:
        main.delete_document("../secrets.txt")
    assert parent.value.status_code == 400
    assert main.sanitize_filename("My Notes.txt") == "My Notes.txt"
    assert list(isolated_storage.iterdir()) == []


@pytest.mark.parametrize("name", ["foo..txt", "..\\secrets.txt", "notes.exe"])
def test_delete_route_rejects_unsafe_names(name, isolated_storage):
    response = client.delete(f"/documents/{name}")
    assert response.status_code == 400
    assert list(isolated_storage.iterdir()) == []


def test_stores_under_generated_id_and_keeps_original_name(isolated_storage):
    text = b"hello from notes"
    response = client.post(
        "/index",
        files={"file": ("notes.txt", text, "text/plain")},
    )
    assert response.status_code == 200
    assert response.json()["filename"] == "notes.txt"

    stored_files = [path for path in isolated_storage.iterdir() if path.name != "manifest.json"]
    assert len(stored_files) == 1
    assert re.fullmatch(r"[a-f0-9]{32}\.txt", stored_files[0].name)
    assert stored_files[0].name != "notes.txt"
    assert stored_files[0].read_bytes() == text

    manifest = json.loads((isolated_storage / "manifest.json").read_text())
    assert manifest["notes.txt"] == stored_files[0].name

    rows = embeddings.collection.get(where={"source": "notes.txt"})
    assert rows["ids"]
    assert {meta["source"] for meta in rows["metadatas"]} == {"notes.txt"}


def test_accepts_pdf_and_rejects_other_types(isolated_storage):
    rejected = client.post(
        "/upload",
        files={"file": ("notes.exe", b"hello", "application/octet-stream")},
    )
    assert rejected.status_code == 400
    assert list(isolated_storage.iterdir()) == []

    buffer = BytesIO()
    writer = PdfWriter()
    writer.add_blank_page(width=72, height=72)
    writer.write(buffer)
    accepted = client.post(
        "/upload",
        files={"file": ("notes.pdf", buffer.getvalue(), "application/pdf")},
    )
    assert accepted.status_code == 200
    assert accepted.json()["filename"] == "notes.pdf"
    stored = [path.name for path in isolated_storage.iterdir() if path.name != "manifest.json"]
    assert len(stored) == 1
    assert stored[0].endswith(".pdf")


def test_rejects_empty_and_oversize_uploads(isolated_storage, monkeypatch):
    empty = client.post("/upload", files={"file": ("notes.txt", b"", "text/plain")})
    assert empty.status_code == 400
    assert list(isolated_storage.iterdir()) == []

    monkeypatch.setattr(main, "MAX_UPLOAD_BYTES", 8)
    oversize = client.post(
        "/index",
        files={"file": ("notes.txt", b"0123456789", "text/plain")},
    )
    assert oversize.status_code == 413
    assert list(isolated_storage.iterdir()) == []


def test_reindex_replaces_chunks_instead_of_duplicating(isolated_storage):
    body = ("alpha " * 200).encode()
    files = {"file": ("notes.txt", body, "text/plain")}
    first = client.post("/index", files=files)
    second = client.post("/index", files=files)
    assert first.status_code == 200
    assert second.status_code == 200

    chunks = chunk_text(body.decode())
    assert len(chunks) > 1
    rows = embeddings.collection.get(where={"source": "notes.txt"})
    assert len(rows["ids"]) == len(chunks)
    assert set(rows["ids"]) == {chunk_id("notes.txt", index) for index in range(len(chunks))}

    shortened = client.post(
        "/index",
        files={"file": ("notes.txt", b"short note", "text/plain")},
    )
    assert shortened.status_code == 200
    replaced = embeddings.collection.get(where={"source": "notes.txt"})
    assert replaced["ids"] == [chunk_id("notes.txt", 0)]
    assert replaced["documents"] == ["short note"]

    stored_files = [path for path in isolated_storage.iterdir() if path.name != "manifest.json"]
    assert len(stored_files) == 1
    assert stored_files[0].read_bytes() == b"short note"


def test_delete_removes_only_matching_source(isolated_storage):
    alpha = client.post("/index", files={"file": ("alpha.txt", b"alpha document", "text/plain")})
    beta = client.post("/index", files={"file": ("beta.txt", b"beta document", "text/plain")})
    assert alpha.status_code == 200
    assert beta.status_code == 200

    manifest = json.loads((isolated_storage / "manifest.json").read_text())
    delete_calls = []
    original_delete = embeddings.collection.delete
    original_get = embeddings.collection.get

    def wrapped_delete(*args, **kwargs):
        delete_calls.append(kwargs)
        return original_delete(*args, **kwargs)

    def wrapped_get(*args, **kwargs):
        raise AssertionError(f"delete must not scan with collection.get ({kwargs})")

    embeddings.collection.delete = wrapped_delete
    embeddings.collection.get = wrapped_get
    try:
        response = client.delete("/documents/alpha.txt")
    finally:
        embeddings.collection.delete = original_delete
        embeddings.collection.get = original_get

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "success"
    assert body["deleted_count"] == 1
    assert body["file_deleted"] is True
    assert delete_calls == [{"where": {"source": "alpha.txt"}}]

    assert not (isolated_storage / manifest["alpha.txt"]).exists()
    assert (isolated_storage / manifest["beta.txt"]).is_file()
    remaining = embeddings.collection.get()
    assert {meta["source"] for meta in remaining["metadatas"]} == {"beta.txt"}

    missing = client.delete("/documents/missing.txt")
    assert missing.status_code == 200
    assert missing.json()["status"] == "not_found"
    assert missing.json()["deleted_count"] == 0
