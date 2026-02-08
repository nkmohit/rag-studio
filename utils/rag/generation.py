from google import genai
import os
from typing import List

api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    raise RuntimeError("GEMINI_API_KEY is not set")

client = genai.Client(api_key=api_key)

MODEL_NAME = "gemini-flash-lite-latest"


def generate_answer(
    query: str,
    contexts: List[str]
) -> str:
    """
    Generate an answer using retrieved document context.
    If no context is provided, generates a general answer.
    """
    if not contexts or all(not c.strip() for c in contexts):
        # No context available - answer as a general assistant
        prompt = f"""
You are a helpful assistant.
Answer the following question to the best of your knowledge.
Be concise and accurate.

Question:
{query}

Answer:
"""
    else:
        # Context available - use RAG approach
        context_text = "\n\n".join(contexts)
        prompt = f"""
You are a helpful assistant.
Answer the question using the context below.
If the context doesn't contain relevant information, you may use your general knowledge.
Be clear about whether you're answering from the context or from general knowledge.

Context:
{context_text}

Question:
{query}

Answer:
"""

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt
    )

    return response.text.strip()
