"""Chunking, embedding, and retrieval over a notes file."""
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np

_model = None

def _get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer

        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model


def chunk_text(text: str, target_words: int = 180, overlap_words: int = 30) -> list[str]:
    """Split into overlapping chunks of ~target_words, packing paragraphs."""
    if type(target_words) is not int or type(overlap_words) is not int or not 0 <= overlap_words < target_words:
        raise ValueError("Chunk size must be positive; overlap must be between zero and chunk size minus one.")
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
            overlap = current[-overlap_words:] if current and overlap_words else []
            current = overlap + words
        # Bound long paragraphs instead of silently relying on model truncation.
        while len(current) > target_words:
            chunks.append(" ".join(current[:target_words]))
            current = current[target_words - overlap_words:]
    if current:
        chunks.append(" ".join(current))
    return chunks


class NotesIndex:
    def __init__(self, chunks: list[str], embeddings: "np.ndarray"):
        self.chunks = chunks
        self.embeddings = embeddings  # (N, dim), L2-normalized

    @classmethod
    def build(cls, text: str) -> "NotesIndex":
        import numpy as np

        chunks = chunk_text(text)
        if not chunks:
            raise ValueError("Notes are empty. Add some study material before building the index.")
        model = _get_model()
        embs = model.encode(chunks, normalize_embeddings=True)
        return cls(chunks, np.asarray(embs, dtype=np.float32))

    def search(self, query: str, k: int = 5) -> list[str]:
        import numpy as np

        if type(k) is not int or k < 1:
            raise ValueError("k must be a positive integer.")
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
    underlined: list[str] = []
    lines = text.splitlines()
    for i, line in enumerate(lines):
        s = line.strip()
        m = re.match(r"^(\d+)\.\s+(.+)$", s)
        if m:
            title = m.group(2).strip()
            if title and len(title) <= 100:
                next_line = lines[i + 1].strip() if i + 1 < len(lines) else ""
                if re.fullmatch(r"[-=]{3,}", next_line):
                    underlined.append(title)
                elif (i == 0 or not lines[i - 1].strip()) and not re.match(r"^\d+\.\s", next_line):
                    titles.append(title)
    # Underlined headings disambiguate sections from numbered lists within them.
    return list(dict.fromkeys(underlined or titles))
