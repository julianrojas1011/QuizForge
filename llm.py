"""Thin wrapper around the Anthropic client."""
import os
from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()

MODEL = "claude-sonnet-4-5-20250929"
_client = Anthropic()


def complete(system: str, user: str, max_tokens: int = 2048) -> str:
    """Single-turn completion. Returns text content."""
    resp = _client.messages.create(
        model=MODEL,
        max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    return resp.content[0].text
