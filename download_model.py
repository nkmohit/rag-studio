#!/usr/bin/env python3
"""
Download embedding model for RAG Studio.
This script downloads the sentence-transformers model to the local models directory.
"""

import os
from sentence_transformers import SentenceTransformer

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_DIR = "./utils/models/retriever"

def download_model():
    """
    Download the embedding model to the local models directory.
    """
    print(f"Downloading model: {MODEL_NAME}")
    print(f"Target directory: {MODEL_DIR}")
    
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    print("This may take a few minutes depending on your internet connection...")
    model = SentenceTransformer(MODEL_NAME)
    
    print(f"\nSaving model to: {MODEL_DIR}")
    model.save(MODEL_DIR)
    
    print("\nModel downloaded successfully!")
    print(f"Model location: {os.path.abspath(MODEL_DIR)}")

if __name__ == "__main__":
    download_model()
