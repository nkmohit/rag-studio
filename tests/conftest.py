"""
Mock Gemini and the local embedding model before the app imports them.
Chroma uses an in-memory client so tests do not need a downloaded model or an API key.
"""
import os
import sys
import types
from unittest.mock import MagicMock

import pytest

os.environ.setdefault("GEMINI_API_KEY", "test-key")
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")


class _EmbeddingMatrix:
    def __init__(self, rows):
        self._rows = rows

    def tolist(self):
        return self._rows


class _Encoder:
    def __init__(self, *args, **kwargs):
        pass

    def encode(self, texts, **kwargs):
        rows = [
            [float((len(text) + index) % 7), 0.1, 0.2]
            for index, text in enumerate(texts)
        ]
        return _EmbeddingMatrix(rows)


sentence_transformers = types.ModuleType("sentence_transformers")
sentence_transformers.SentenceTransformer = _Encoder
sys.modules["sentence_transformers"] = sentence_transformers

import chromadb

_chroma = chromadb.EphemeralClient()


def _persistent_client(*args, **kwargs):
    return _chroma


chromadb.PersistentClient = _persistent_client

genai = types.ModuleType("google.genai")
genai.Client = MagicMock(name="GenAIClient")
sys.modules["google.genai"] = genai
import google

google.genai = genai

import main
from utils.rag import embeddings


@pytest.fixture(autouse=True)
def isolated_storage(tmp_path, monkeypatch):
    monkeypatch.setattr(main, "UPLOAD_DIR", str(tmp_path))
    try:
        embeddings.chroma_client.delete_collection("documents")
    except Exception:
        pass
    embeddings.collection = embeddings.chroma_client.get_or_create_collection(name="documents")
    yield tmp_path
