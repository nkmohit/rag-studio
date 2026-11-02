from google import genai
import logging
import os
from typing import List

logger = logging.getLogger(__name__)

api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    raise RuntimeError("GEMINI_API_KEY is not set")

client = genai.Client(api_key=api_key)

MODEL_NAME = "gemini-flash-lite-latest"


def rewrite_query_multi_perspective(query: str) -> List[str]:
    """
    Rewrite the user's query into multiple perspectives to improve retrieval.
    Returns list of query variations including the original.
    """
    prompt = f"""
You are an expert at understanding user intent and rewriting queries for better document retrieval.

Given a user's question, generate 2-3 alternative phrasings that capture the same intent but use different words or perspectives.

Rules:
1. Keep the core intent identical
2. Use synonyms and alternative terminology
3. Vary the question structure (e.g., "What is X?" vs "Explain X" vs "How does X work?")
4. Keep each rewrite concise (similar length to original)

Original Question: {query}

Provide ONLY the alternative questions, one per line, without numbering or explanation.
"""

    try:
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt
        )
        
        rewrites = [q.strip() for q in response.text.strip().split('\n') if q.strip()]
        all_queries = [query] + rewrites[:3]
        
        return all_queries
    except Exception:
        logger.exception("Query rewriting failed")
        return [query]


def generate_answer(
    query: str,
    contexts: List[str],
    has_relevant_context: bool = False
) -> str:
    """
    Generate an answer ONLY if relevant context is provided.
    Enforces strict grounding and prevents hallucination.
    """
    if not has_relevant_context or not contexts or all(not c.strip() for c in contexts):
        return "I couldn't find relevant information in the uploaded documents to answer this question. Please upload documents related to your query or rephrase your question."
    
    context_text = "\n\n".join(contexts)
    prompt = f"""
You are a precise assistant that answers questions STRICTLY based on provided context.

CRITICAL RULES:
1. Answer ONLY using information from the context below
2. If the context doesn't contain the answer, say "The provided documents don't contain this information"
3. Do NOT use external knowledge or make assumptions
4. Quote or reference specific parts of the context when answering

Context from uploaded documents:
{context_text}

Question:
{query}

Answer (based strictly on the context above):
"""

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt
    )

    return response.text.strip()
