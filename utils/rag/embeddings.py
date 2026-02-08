from sentence_transformers import SentenceTransformer
import chromadb
from typing import List
import uuid


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
    except Exception as e:
        print(f"Error getting documents: {e}")
        return []


def delete_document_by_source(source: str):
    """
    Delete all chunks belonging to a specific document source.
    """
    try:
        all_data = collection.get()
        
        ids_to_delete = []
        if all_data and all_data.get("ids") and all_data.get("metadatas"):
            for idx, metadata in enumerate(all_data["metadatas"]):
                if metadata and metadata.get("source") == source:
                    ids_to_delete.append(all_data["ids"][idx])
        
        if ids_to_delete:
            collection.delete(ids=ids_to_delete)
            return {"status": "success", "deleted_count": len(ids_to_delete)}
        else:
            return {"status": "not_found", "deleted_count": 0}
    except Exception as e:
        return {"status": "error", "message": str(e)}