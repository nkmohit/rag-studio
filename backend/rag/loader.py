from pypdf import PdfReader


def extract_text_from_file(path: str) -> str:
    """
    Extract text from txt or pdf files.
    """
    if path.endswith(".txt"):
        with open(path, "r", encoding="utf-8") as f:
            return f.read()

    if path.endswith(".pdf"):
        reader = PdfReader(path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""
        return text

    return ""
