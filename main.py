import json
import logging
import os
import re
import uuid

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

from fastapi import FastAPI, UploadFile, HTTPException
from utils.rag.loader import extract_text_from_file
from utils.rag.embeddings import index_text, query_text_multi_query, get_all_documents, delete_document_by_source
from utils.rag.generation import generate_answer, rewrite_query_multi_perspective
from pydantic import BaseModel

app = FastAPI()

UPLOAD_DIR = "utils/uploads"
MAX_UPLOAD_MB = 10
MAX_UPLOAD_BYTES = MAX_UPLOAD_MB * 1024 * 1024
ALLOWED_EXTENSIONS = {".pdf", ".txt"}
# Basename only: letters, digits, space, dot, underscore, hyphen.
_SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9 ._-]{0,127}$")
_STORED_ID = re.compile(r"^[a-f0-9]{32}\.(pdf|txt)$")
_MANIFEST_NAME = "manifest.json"

os.makedirs(UPLOAD_DIR, exist_ok=True)


class QueryRequest(BaseModel):
    query: str
    top_k: int = 5
    generate: bool = True


def sanitize_filename(filename: str) -> str:
    """
    Accept one safe basename, or reject path traversal and unexpected types.
    """
    if not isinstance(filename, str) or not filename:
        raise HTTPException(status_code=400, detail="invalid filename")
    if (
        filename != os.path.basename(filename)
        or ".." in filename
        or "/" in filename
        or "\\" in filename
        or "\x00" in filename
    ):
        raise HTTPException(status_code=400, detail="invalid filename")
    if not _SAFE_NAME.fullmatch(filename):
        raise HTTPException(status_code=400, detail="invalid filename")
    extension = os.path.splitext(filename)[1].lower()
    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail="only PDF and TXT files are allowed")
    return filename


def _upload_root() -> str:
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    return os.path.realpath(UPLOAD_DIR)


def _manifest_path() -> str:
    return os.path.join(_upload_root(), _MANIFEST_NAME)


def _load_manifest() -> dict:
    path = _manifest_path()
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise HTTPException(status_code=500, detail="invalid upload manifest")
    return data


def _save_manifest(manifest: dict) -> None:
    path = _manifest_path()
    temporary = f"{path}.tmp"
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle)
    os.replace(temporary, path)


def _path_for_stored(stored_name: str) -> str:
    if not _STORED_ID.fullmatch(stored_name):
        raise HTTPException(status_code=500, detail="invalid stored file id")
    root = _upload_root()
    path = os.path.realpath(os.path.join(root, stored_name))
    if os.path.commonpath([root, path]) != root:
        raise HTTPException(status_code=400, detail="invalid filename")
    return path


def _remove_stored(stored_name: str) -> bool:
    path = _path_for_stored(stored_name)
    if os.path.isfile(path):
        os.remove(path)
        return True
    return False


def _store_upload(safe_name: str, content: bytes) -> str:
    """
    Write bytes under a generated id. The caller's filename is manifest metadata.
    """
    extension = os.path.splitext(safe_name)[1].lower()
    stored_name = f"{uuid.uuid4().hex}{extension}"
    path = _path_for_stored(stored_name)
    manifest = _load_manifest()
    previous = manifest.get(safe_name)
    with open(path, "wb") as handle:
        handle.write(content)
    if isinstance(previous, str) and previous != stored_name:
        try:
            _remove_stored(previous)
        except (OSError, HTTPException):
            logger.warning("Could not remove previous upload for %s", safe_name)
    manifest[safe_name] = stored_name
    _save_manifest(manifest)
    return path


async def _accept_upload(file: UploadFile) -> tuple[str, str]:
    safe_name = sanitize_filename(file.filename or "")
    content = await file.read(MAX_UPLOAD_BYTES + 1)
    if not content:
        raise HTTPException(status_code=400, detail="empty file")
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"file exceeds {MAX_UPLOAD_MB} MB limit",
        )
    return safe_name, _store_upload(safe_name, content)


@app.get("/")
def read_root():
    return {"message": "RAG App Started"}


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.get("/documents")
def list_documents():
    """
    Get a list of all indexed documents.
    """
    documents = get_all_documents()
    return {"documents": documents}


@app.delete("/documents/{filename}")
def delete_document(filename: str):
    """
    Delete a document and all its chunks from the vector store.
    Also removes the uploaded file.
    """
    safe_name = sanitize_filename(filename)
    result = delete_document_by_source(safe_name)
    manifest = _load_manifest()
    stored_name = manifest.get(safe_name)
    if not isinstance(stored_name, str):
        return result
    try:
        result["file_deleted"] = _remove_stored(stored_name)
    except (OSError, HTTPException):
        logger.exception("Failed to delete stored upload %s", safe_name)
        result["file_delete_error"] = "failed to delete file"
        return result
    manifest.pop(safe_name, None)
    _save_manifest(manifest)
    return result


@app.post("/upload")
async def upload_file(file: UploadFile):
    """
    Accept a file and save it locally.
    """
    safe_name, path = await _accept_upload(file)
    text = extract_text_from_file(path)
    return {
        "filename": safe_name,
        "chars": len(text),
    }


@app.post("/index")
async def index_file(file: UploadFile):
    """
    Full indexing pipeline: upload, extract, chunk, embed, store.
    """
    safe_name, path = await _accept_upload(file)
    text = extract_text_from_file(path)

    if not text.split():
        return {"status": "error", "message": "no text extracted"}

    index_text(text=text, source=safe_name)

    return {
        "status": "indexed",
        "filename": safe_name,
    }


@app.post("/query")
def query_docs(request: QueryRequest):
    """
    RAG query endpoint with multi-query retrieval and strict grounding.

    Features:
    - Multi-query search with automatic query rewriting
    - Similarity threshold filtering (only applied when >5 documents)
    - Strict grounding - only answers with relevant context
    - Source attribution
    """
    documents = []
    sources_used = []
    has_relevant_context = False
    queries_used = []

    all_docs = get_all_documents()
    if not all_docs:
        return {
            "query": request.query,
            "retrieved_chunks": [],
            "sources_used": [],
            "has_relevant_context": False,
            "queries_used": [request.query],
            "answer": "No documents have been uploaded yet. Please upload documents first to ask questions about them."
        }

    logger.info("Original query: %s", request.query)
    try:
        queries_used = rewrite_query_multi_perspective(request.query)
        logger.info("Rewritten queries: %s", queries_used)
    except Exception:
        logger.exception("Query rewriting failed")
        queries_used = [request.query]

    apply_similarity_threshold = len(all_docs) > 5
    threshold = 0.5 if apply_similarity_threshold else 0.0

    logger.info(
        "Total documents: %s, Applying threshold: %s (threshold=%s)",
        len(all_docs),
        apply_similarity_threshold,
        threshold,
    )

    try:
        results = query_text_multi_query(
            queries=queries_used,
            top_k=request.top_k,
            similarity_threshold=threshold,
            apply_threshold=apply_similarity_threshold
        )

        documents = results.get("documents", [[]])[0]
        has_relevant_context = results.get("has_relevant_context", False)

        metadatas = results.get("metadatas", [[]])[0]
        sources_used = list(set([m.get("source", "unknown") for m in metadatas if m]))

        logger.info(
            "Retrieved %s relevant chunks from sources: %s",
            len(documents),
            sources_used,
        )

    except Exception:
        logger.exception("Retrieval error")
        documents = []
        has_relevant_context = False

    response = {
        "query": request.query,
        "retrieved_chunks": documents,
        "sources_used": sources_used,
        "has_relevant_context": has_relevant_context,
        "queries_used": queries_used
    }

    if request.generate:
        answer = generate_answer(
            query=request.query,
            contexts=documents,
            has_relevant_context=has_relevant_context
        )
        response["answer"] = answer
    else:
        if not has_relevant_context:
            response["answer"] = "No relevant context found in uploaded documents."

    return response
