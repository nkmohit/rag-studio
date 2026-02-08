# RAG Studio

A demo implementation of a Retrieval-Augmented Generation (RAG) system that allows users to upload documents, ask questions, and see source attribution for generated answers.

This is a **reference implementation** for learning and demonstration purposes, not a production-ready application.

---

## What is RAG Studio?

RAG Studio demonstrates the core RAG pipeline:

1. **Upload** PDF or TXT documents
2. **Index** documents using semantic embeddings
3. **Query** documents with natural language questions
4. **Generate** answers grounded in retrieved document chunks
5. **Attribute** answers by showing the exact source chunks used

The system emphasizes **source transparency** by always showing which document chunks were used to generate each answer.

---

## Features

- Document upload (PDF, TXT)
- Semantic search with ChromaDB vector database
- Answer generation using Google Gemini
- Source chunk attribution for every answer
- Single-document sessions (one document per session)
- Clean separation between retrieval and generation

---

## Architecture

**Backend (FastAPI):**
- `/upload` - Accept and store document files
- `/index` - Extract text, chunk, embed, and store in vector database
- `/query` - Retrieve relevant chunks and generate answers

**Components:**
- **SentenceTransformers** for embeddings (local model at `./models/retriever`)
- **ChromaDB** for vector storage (persistent at `./chroma`)
- **Google Gemini** for answer generation
- **Streamlit** frontend as demo interface

**Pipeline:**
```
Upload → Extract Text → Chunk (500 chars, 50 overlap) → 
Embed → Store in ChromaDB → Query → Retrieve → Generate Answer
```

---

## How to Run Locally

### Prerequisites

- Python 3.8+
- Google Gemini API key ([Get one here](https://makersuite.google.com/app/apikey))

### Backend Setup

```bash
cd backend
pip install -r requirements.txt

# Create .env file with your API key
echo "GEMINI_API_KEY=your_api_key_here" > .env

# Start the backend server
uvicorn main:app --reload
```

The backend will run on `http://localhost:8000`

### Frontend Setup

In a separate terminal:

```bash
# Install frontend dependencies
pip install -r requirements.txt

# Start the Streamlit app
streamlit run streamlit_app.py
```

The Streamlit UI will open automatically in your browser (usually `http://localhost:8501`)

### Usage

1. Upload a PDF or TXT file using the file uploader
2. Wait for the "Indexed successfully" message
3. Ask a question about your document
4. View the generated answer and retrieved source chunks

---

## Out of Scope (V1)

This demo intentionally excludes:

- Multi-document support
- Chat history or conversation memory
- User authentication
- Multiple LLM providers
- Custom chunking strategies
- Deployment configuration
- Production-grade error handling
- File management features
- Performance optimization

---

## Project Status

**Version 1: Demo-Ready**

This is an educational reference implementation demonstrating core RAG concepts. It is **not intended for production use**.

The codebase prioritizes clarity and simplicity over performance, scalability, or robustness.

---

## Design Principles

- Minimal but correct RAG pipeline
- Clear separation of concerns
- Explicit source attribution
- No unnecessary abstractions
- Single responsibility per component

---

## License

MIT License - See [LICENSE](LICENSE) file for details.  
