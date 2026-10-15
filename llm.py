"""Thin wrapper around the Anthropic client."""
import os

DEFAULT_MODEL = "claude-sonnet-4-5-20250929"
_client = None


def configure() -> str:
    """Load local settings and fail before downloading an embedding model."""
    try:
        from dotenv import load_dotenv
    except ImportError as exc:
        raise RuntimeError("Install the runtime dependencies: pip install -r requirements.txt") from exc
    load_dotenv()
    key = os.getenv("ANTHROPIC_API_KEY", "").strip()
    if not key or key == "your-key-here":
        raise ValueError("Set ANTHROPIC_API_KEY in .env before starting a live quiz. Try python demo.py for the offline example.")
    return os.getenv("ANTHROPIC_MODEL", "").strip() or DEFAULT_MODEL


def get_client():
    global _client
    configure()
    if _client is None:
        from anthropic import Anthropic

        _client = Anthropic(timeout=60.0, max_retries=2)
    return _client


def complete(system: str, user: str, max_tokens: int = 2048) -> str:
    """Single-turn completion. Returns text content."""
    resp = get_client().messages.create(
        model=configure(),
        max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    if resp.stop_reason != "end_turn":
        raise ValueError(f"Model response was incomplete ({resp.stop_reason}); try fewer questions.")
    text = "\n".join(block.text for block in resp.content if getattr(block, "type", None) == "text").strip()
    if not text:
        raise ValueError("Model returned no text.")
    return text
