import os
from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, UploadFile, File, HTTPException
from utils.rag.loader import extract_text_from_file
from utils.rag.embeddings import index_text, query_text, get_all_documents, delete_document_by_source
from utils.rag.generation import generate_answer
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
    # Delete from vector store
    result = delete_document_by_source(filename)
    
    # Delete physical file
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
    STEP 1 of Ingestion pipeline.
    
    :param file: File to be embeded
    :type file: UploadFile
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
    Full indexing pipeline:
    upload → extract → chunk → embed → store
    
    :param file: Description
    :type file: UploadFile
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
    RAG query endpoint:
    - semantic retrieval
    - optional generation
    
    Works even if no documents are indexed (will generate without context).
    """
    documents = []
    sources_used = []
    
    # Try to retrieve documents if any exist
    try:
        results = query_text(
            query=request.query,
            top_k=request.top_k
        )
        documents = results.get("documents", [[]])[0]
        
        # Extract sources from metadata
        metadatas = results.get("metadatas", [[]])[0]
        sources_used = list(set([m.get("source", "unknown") for m in metadatas if m]))
    except Exception as e:
        print(f"Retrieval error: {e}")
        documents = []

    response = {
        "query": request.query,
        "retrieved_chunks": documents,
        "sources_used": sources_used
    }

    if request.generate:
        answer = generate_answer(
            query=request.query,
            contexts=documents
        )
        response["answer"] = answer

    return response
