import os
from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, UploadFile, File, HTTPException
from utils.rag.loader import extract_text_from_file
from utils.rag.embeddings import index_text, query_text_multi_query, get_all_documents, delete_document_by_source
from utils.rag.generation import generate_answer, rewrite_query_multi_perspective
from pydantic import BaseModel

app = FastAPI()

UPLOAD_DIR = "utils/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

class QueryRequest(BaseModel):
    query: str
    top_k: int = 5
    generate: bool = True


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
    result = delete_document_by_source(filename)
    
    file_path = os.path.join(UPLOAD_DIR, filename)
    if os.path.exists(file_path):
        try:
            os.remove(file_path)
            result["file_deleted"] = True
        except Exception as e:
            result["file_delete_error"] = str(e)
    
    return result

@app.post("/upload")
async def upload_file(file : UploadFile):
    """
    Accept a file and save it locally.
    """
    file_path = os.path.join(UPLOAD_DIR, file.filename)

    with open(file_path, "wb") as f:
        content = await file.read()
        f.write(content)
    
    text = extract_text_from_file(file_path)

    return {
        "filename": file.filename,
        "chars": len(text)
    }

@app.post("/index")
async def index_file(file : UploadFile):
    """
    Full indexing pipeline: upload, extract, chunk, embed, store.
    """
    file_path = os.path.join(UPLOAD_DIR, file.filename)

    with open(file_path, "wb") as f:
        content = await file.read()
        f.write(content)
    
    text = extract_text_from_file(file_path)

    if not text.split():
        return {"status": "error", "message": "no text extracted" }
    
    index_text(text=text, source= file.filename)

    return {
        "status": "indexed",
        "filename": file.filename
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
    
    print(f"Original query: {request.query}")
    try:
        queries_used = rewrite_query_multi_perspective(request.query)
        print(f"Rewritten queries: {queries_used}")
    except Exception as e:
        print(f"Query rewriting failed: {e}")
        queries_used = [request.query]
    
    apply_similarity_threshold = len(all_docs) > 5
    threshold = 0.5 if apply_similarity_threshold else 0.0
    
    print(f"Total documents: {len(all_docs)}, Applying threshold: {apply_similarity_threshold} (threshold={threshold})")
    
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
        
        print(f"Retrieved {len(documents)} relevant chunks from sources: {sources_used}")
        
    except Exception as e:
        print(f"Retrieval error: {e}")
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
