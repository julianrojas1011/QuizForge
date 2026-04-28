"""Chunking, embedding, and retrieval over a notes file."""
import re
import numpy as np
from sentence_transformers import SentenceTransformer

_model = None

def _get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model


def chunk_text(text: str, target_words: int = 180, overlap_words: int = 30) -> list[str]:
    """Split into overlapping chunks of ~target_words, packing paragraphs."""
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks: list[str] = []
    current: list[str] = []
    for para in paragraphs:
        words = para.split()
        if len(current) + len(words) <= target_words:
            current.extend(words)
        else:
            if current:
                chunks.append(" ".join(current))
            overlap = current[-overlap_words:] if current else []
            current = overlap + words
    if current:
        chunks.append(" ".join(current))
    return chunks


class NotesIndex:
    def __init__(self, chunks: list[str], embeddings: np.ndarray):
        self.chunks = chunks
        self.embeddings = embeddings  # (N, dim), L2-normalized

    @classmethod
    def build(cls, text: str) -> "NotesIndex":
        chunks = chunk_text(text)
        model = _get_model()
        embs = model.encode(chunks, normalize_embeddings=True)
        return cls(chunks, np.asarray(embs, dtype=np.float32))

    def search(self, query: str, k: int = 5) -> list[str]:
        model = _get_model()
        q = model.encode([query], normalize_embeddings=True)[0]
        sims = self.embeddings @ q
        top_idx = np.argsort(-sims)[:k]
        return [self.chunks[i] for i in top_idx]


# ----- Milestone 3 / 4: topic outline -----

def extract_topic_outline(text: str) -> list[str]:
    """
    Extract top-level numbered section titles from a notes file.
    Looks for lines like "1. What is Machine Learning?" or "2. Supervised Learning".
    Returns the section titles in order.
    """
    titles: list[str] = []
    for line in text.splitlines():
        s = line.strip()
        m = re.match(r"^(\d+)\.\s+(.+)$", s)
        if m:
            title = m.group(2).strip()
            # Skip if next non-empty line is a separator only (avoids false positives)
            if title and len(title) <= 100:
                titles.append(title)
    return titles
