from sentence_transformers import SentenceTransformer
import chromadb
from typing import List
import uuid


# Embedding model initialization
# The model is loaded from a local path to avoid network dependency and reduce startup variability.

model = SentenceTransformer(model_name_or_path="./models/retriever")

# Initialize Chroma client (local persistent DB)
chroma_client = chromadb.PersistentClient(
    path="./chroma"
)

# Single collection for now
collection = chroma_client.get_or_create_collection(
    name="documents"
)


def chunk_text(
    text: str,
    chunk_size: int = 500,
    overlap: int = 50
) -> List[str]:
    """
    Simple sliding window chunking.
    """
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap

    return chunks


def index_text(text: str, source: str):
    """
    Chunk text, embed chunks, and store in Chroma.
    """
    chunks = chunk_text(text)

    embeddings = model.encode(chunks).tolist()

    ids = [str(uuid.uuid4()) for _ in chunks]

    metadatas = [{"source": source} for _ in chunks]

    collection.add(
        documents=chunks,
        embeddings=embeddings,
        metadatas=metadatas,
        ids=ids
    )

def query_text(
    query: str,
    top_k: int = 5
):
    """
    Perform semantic similarity search against the vector store.

    Steps:
    - embed the query text
    - run vector similarity search in Chroma
    - return top matching document chunks
    """
    query_embedding = model.encode([query]).tolist()

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=top_k
    )

    return results