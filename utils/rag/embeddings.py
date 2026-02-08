from sentence_transformers import SentenceTransformer
import chromadb
from typing import List
import uuid


# Embedding model initialization
# The model is loaded from a local path to avoid network dependency and reduce startup variability.

model = SentenceTransformer(model_name_or_path="./utils/models/retriever")

# Initialize Chroma client (local persistent DB)
chroma_client = chromadb.PersistentClient(
    path="./utils/chroma"
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


def get_all_documents():
    """
    Get all unique document sources from the collection.
    """
    try:
        # Get all items from collection
        all_data = collection.get()
        
        # Extract unique sources
        sources = set()
        if all_data and all_data.get("metadatas"):
            for metadata in all_data["metadatas"]:
                if metadata and "source" in metadata:
                    sources.add(metadata["source"])
        
        return list(sources)
    except Exception as e:
        print(f"Error getting documents: {e}")
        return []


def delete_document_by_source(source: str):
    """
    Delete all chunks belonging to a specific document source.
    """
    try:
        # Get all items
        all_data = collection.get()
        
        # Find IDs that match the source
        ids_to_delete = []
        if all_data and all_data.get("ids") and all_data.get("metadatas"):
            for idx, metadata in enumerate(all_data["metadatas"]):
                if metadata and metadata.get("source") == source:
                    ids_to_delete.append(all_data["ids"][idx])
        
        # Delete matching IDs
        if ids_to_delete:
            collection.delete(ids=ids_to_delete)
            return {"status": "success", "deleted_count": len(ids_to_delete)}
        else:
            return {"status": "not_found", "deleted_count": 0}
    except Exception as e:
        return {"status": "error", "message": str(e)}