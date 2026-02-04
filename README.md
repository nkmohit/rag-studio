# RAG Studio

RAG Studio is a web-based Retrieval-Augmented Generation (RAG) platform that allows users to upload documents, query them using natural language, and see exactly which document chunks were used to generate each answer.

The project emphasizes **grounded generation**, **source transparency**, and **clean system design**.

---

## Motivation

Large language models are powerful but unreliable without grounding.  
RAG Studio demonstrates how to build a production-style RAG system that minimizes hallucination by explicitly retrieving and attributing source content.

---

## Current Status

**Version 1 — In Active Development**  

---

## Versioned Roadmap

- **Version 1 (Feb 2026)**  
  Web-based RAG platform using a single LLM provider (Gemini), focused on correctness, attribution, and usability.

- **Version 2 (Future)**  
  Extensible RAG platform with multiple LLM providers and pluggable retrieval strategies.

- **Version 3 (Future)**  
  Desktop application with local document processing and local LLM support.

---

## Planned Tech Stack (Version 1)

### Backend
- FastAPI
- Sentence Transformers (embeddings)
- ChromaDB (vector storage)
- Google Gemini (user-provided API key)

### Frontend
- Next.js
- React
- Tailwind CSS

---

## Design Principles

- Minimal but correct RAG pipeline
- Clear separation of concerns
- Explicit source attribution
- No unnecessary abstractions

---

## Status

This repository is under active development.  
