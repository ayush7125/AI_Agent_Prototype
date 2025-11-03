import re
import json
from typing import List, Dict


# ============================================================
# 🔹 Text Cleaning
# ============================================================
def clean_text(s: str) -> str:
    """
    Cleans scientific text for summarization.
    - Removes excessive whitespace
    - Strips citation markers like [1], (Smith et al., 2021)
    - Fixes broken lines and OCR artifacts
    """
    if not s or not isinstance(s, str):
        return ""

    # Remove citation markers and references
    s = re.sub(r"\[\d+(,\s*\d+)*\]", "", s)  # [1], [2,3]
    s = re.sub(r"\([A-Z][a-z]+ et al\., \d{4}\)", "", s)  # (Smith et al., 2020)
    s = re.sub(r"\b(et al\.|ibid\.|op\.cit\.)\b", "", s, flags=re.IGNORECASE)

    # Remove LaTeX/math snippets
    s = re.sub(r"\$.*?\$", "", s)
    s = re.sub(r"\\begin\{.*?\}.*?\\end\{.*?\}", "", s, flags=re.DOTALL)

    # Fix line breaks and redundant spaces
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"([a-z])-\s+([a-z])", r"\1\2", s)  # join hyphenated words

    return s.strip()


# ============================================================
# 🔹 Text Chunking
# ============================================================
def chunk_text(text: str, chunk_size: int = 1500, overlap: int = 150) -> List[str]:
    """
    Splits long text into overlapping, sentence-aware chunks.
    Preserves sentence boundaries for coherent model input.
    """
    if not text or not isinstance(text, str):
        return []

    # Split text into sentences
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks, current_chunk = [], ""

    for sentence in sentences:
        # Clean sentence
        sentence = clean_text(sentence)
        if not sentence:
            continue

        # Add sentence to chunk
        if len(current_chunk) + len(sentence) < chunk_size:
            current_chunk += sentence + " "
        else:
            chunks.append(current_chunk.strip())
            # Overlap control: start next chunk with trailing portion
            current_chunk = " ".join(current_chunk.split()[-overlap:]) + " " + sentence + " "

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks


# ============================================================
# 🔹 JSON I/O Helpers
# ============================================================
def save_json(path: str, obj: Dict):
    """
    Save dictionary as JSON file with UTF-8 encoding.
    """
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_json(path: str) -> Dict:
    """
    Load and parse a JSON file safely.
    """
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)
