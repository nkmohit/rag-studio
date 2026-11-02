from sentence_transformers import SentenceTransformer
import chromadb
from typing import List
import hashlib
import logging

logger = logging.getLogger(__name__)


model = SentenceTransformer(model_name_or_path="./utils/models/retriever")

chroma_client = chromadb.PersistentClient(path="./utils/chroma")

collection = chroma_client.get_or_create_collection(name="documents")


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


def chunk_id(source: str, index: int) -> str:
    """
    Stable id for one chunk. Re-indexing the same source reuses these ids.
    """
    digest = hashlib.sha256()
    digest.update(source.encode("utf-8"))
    digest.update(b"\0")
    digest.update(str(index).encode("ascii"))
    return digest.hexdigest()


def index_text(text: str, source: str):
    """
    Chunk text, embed chunks, and store in Chroma.
    Existing chunks for this source are removed first so a re-index cannot duplicate them.
    """
    chunks = chunk_text(text)
    if not chunks:
        return
    vectors = model.encode(chunks).tolist()
    ids = [chunk_id(source, index) for index in range(len(chunks))]
    metadatas = [{"source": source} for _ in chunks]

    collection.delete(where={"source": source})
    collection.add(
        documents=chunks,
        embeddings=vectors,
        metadatas=metadatas,
        ids=ids
    )


def query_text(
    query: str,
    top_k: int = 5,
    similarity_threshold: float = 0.5
):
    """
    Perform semantic similarity search against the vector store.
    Returns results filtered by similarity threshold.
    """
    query_embedding = model.encode([query]).tolist()

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=top_k
    )

    filtered_documents = []
    filtered_metadatas = []
    filtered_distances = []
    
    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]
    
    for doc, meta, dist in zip(documents, metadatas, distances):
        similarity = 1 / (1 + dist)
        if similarity >= similarity_threshold:
            filtered_documents.append(doc)
            filtered_metadatas.append(meta)
            filtered_distances.append(dist)
    
    return {
        "documents": [filtered_documents] if filtered_documents else [[]],
        "metadatas": [filtered_metadatas] if filtered_metadatas else [[]],
        "distances": [filtered_distances] if filtered_distances else [[]],
        "has_relevant_context": len(filtered_documents) > 0
    }


def query_text_multi_query(
    queries: List[str],
    top_k: int = 5,
    similarity_threshold: float = 0.5,
    apply_threshold: bool = True
):
    """
    Perform multi-query semantic search using query rewriting.
    
    Improves retrieval by searching with multiple query perspectives,
    aggregating results, deduplicating, and ranking by best score.
    
    Args:
        queries: List of query variations (original + rewrites)
        top_k: Number of results to return
        similarity_threshold: Minimum similarity score (only applied if apply_threshold=True)
        apply_threshold: Whether to filter by similarity threshold
    """
    doc_scores = {}
    
    for query in queries:
        query_embedding = model.encode([query]).tolist()
        
        results = collection.query(
            query_embeddings=query_embedding,
            n_results=top_k
        )
        
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]
        
        for doc, meta, dist in zip(documents, metadatas, distances):
            similarity = 1 / (1 + dist)
            
            should_include = not apply_threshold or similarity >= similarity_threshold
            
            if should_include:
                if doc not in doc_scores:
                    doc_scores[doc] = {
                        "metadata": meta,
                        "min_distance": dist,
                        "count": 1
                    }
                else:
                    doc_scores[doc]["min_distance"] = min(doc_scores[doc]["min_distance"], dist)
                    doc_scores[doc]["count"] += 1
    
    sorted_docs = sorted(
        doc_scores.items(),
        key=lambda x: (x[1]["min_distance"] - (x[1]["count"] * 0.1))
    )
    
    final_documents = [doc for doc, _ in sorted_docs[:top_k]]
    final_metadatas = [data["metadata"] for _, data in sorted_docs[:top_k]]
    final_distances = [data["min_distance"] for _, data in sorted_docs[:top_k]]
    
    return {
        "documents": [final_documents] if final_documents else [[]],
        "metadatas": [final_metadatas] if final_metadatas else [[]],
        "distances": [final_distances] if final_distances else [[]],
        "has_relevant_context": len(final_documents) > 0,
        "queries_used": queries
    }


def get_all_documents():
    """
    Get all unique document sources from the collection.
    """
    try:
        all_data = collection.get()
        
        sources = set()
        if all_data and all_data.get("metadatas"):
            for metadata in all_data["metadatas"]:
                if metadata and "source" in metadata:
                    sources.add(metadata["source"])
        
        return list(sources)
    except Exception:
        logger.exception("Error getting documents")
        return []


def delete_document_by_source(source: str):
    """
    Delete all chunks belonging to a specific document source.
    """
    try:
        result = collection.delete(where={"source": source})
        deleted = result.get("deleted") if isinstance(result, dict) else None
        if deleted == 0:
            return {"status": "not_found", "deleted_count": 0}
        if isinstance(deleted, int):
            return {"status": "success", "deleted_count": deleted}
        return {"status": "success"}
    except Exception as exc:
        logger.exception("Failed to delete document %s", source)
        return {"status": "error", "message": str(exc)}