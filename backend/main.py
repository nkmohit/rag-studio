import os
from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, UploadFile, File
from rag.loader import extract_text_from_file
from rag.embeddings import index_text, query_text
from rag.generation import generate_answer
from pydantic import BaseModel

app = FastAPI()

UPLOAD_DIR = "uploads"
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
        "status": "indexed"
    }

@app.post("/query")
def query_docs(request: QueryRequest):
    """
    RAG query endpoint:
    - semantic retrieval
    - optional generation
    """
    results = query_text(
        query=request.query,
        top_k=request.top_k
    )

    documents = results.get("documents", [[]])[0]

    response = {
        "query": request.query,
        "retrieved_chunks": documents
    }

    if request.generate:
        answer = generate_answer(
            query=request.query,
            contexts=documents
        )
        response["answer"] = answer

    return response
