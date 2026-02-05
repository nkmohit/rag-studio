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
    Generate a grounded answer using retrieved document context.
    """
    context_text = "\n\n".join(contexts)

    prompt = f"""
You are a helpful assistant.
Answer the question using ONLY the context below.
If the answer is not present, say "I don't know".

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
