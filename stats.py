"""Per-topic performance tracking with persistence to performance.json."""
import json
import math
import os
import tempfile
from pathlib import Path

STATS_PATH = Path("performance.json")
DIFFICULTIES = ["easy", "medium", "hard"]
DEFAULT_DIFFICULTY = "easy"


def _empty_topic() -> dict:
    return {
        "attempts": 0,
        "correct": 0,
        "difficulty": DEFAULT_DIFFICULTY,
        "history": [],  # list of per-question floats in [0.0, 1.0]
    }


def load_stats() -> dict:
    if not STATS_PATH.exists():
        return {"topics": {}}
    try:
        with STATS_PATH.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict) or not isinstance(data.get("topics"), dict):
            raise ValueError("Expected an object containing a topics object.")
        for name, topic in data["topics"].items():
            if not isinstance(name, str) or not name.strip() or not isinstance(topic, dict):
                raise ValueError("Invalid topic record.")
            attempts, correct = topic.get("attempts"), topic.get("correct")
            history = topic.get("history")
            if type(attempts) is not int or attempts < 0:
                raise ValueError("Attempts must be a non-negative integer.")
            if not _finite_number(correct) or not 0 <= correct <= attempts:
                raise ValueError("Correct count must be between zero and attempts.")
            if topic.get("difficulty") not in DIFFICULTIES:
                raise ValueError("Unknown saved difficulty.")
            if not isinstance(history, list) or len(history) > min(50, attempts):
                raise ValueError("Invalid score history.")
            if any(not _finite_number(score) or not 0 <= score <= 1 for score in history):
                raise ValueError("History scores must be finite numbers between zero and one.")
            if sum(history) > correct + 1e-9:
                raise ValueError("History does not match the correct count.")
        return data
    except (ValueError, OSError) as exc:
        raise ValueError(f"Cannot read {STATS_PATH}: {exc} Back up and repair or rename this file; it has not been overwritten.") from exc


def _finite_number(value) -> bool:
    return type(value) is int or (type(value) is float and math.isfinite(value))


def save_stats(data: dict) -> None:
    """Replace only after the new record has been fully written."""
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=STATS_PATH.parent, delete=False) as f:
            temporary = Path(f.name)
            json.dump(data, f, indent=2, allow_nan=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temporary, STATS_PATH)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _bump(difficulty: str, direction: int) -> str:
    """direction: +1 up, -1 down, 0 stay. Clamped to bounds."""
    idx = DIFFICULTIES.index(difficulty)
    new_idx = max(0, min(len(DIFFICULTIES) - 1, idx + direction))
    return DIFFICULTIES[new_idx]


def record_results(topic: str, per_question_scores: list[float]) -> dict:
    """
    Record results for one quiz session on `topic`.
    per_question_scores: list of floats in [0,1]. For MCQ: 0 or 1. For open: score/10.
    Returns updated topic dict (already saved).
    """
    if not isinstance(topic, str) or not topic.strip():
        raise ValueError("A non-empty topic is required.")
    if not per_question_scores or any(not _finite_number(score) or not 0 <= score <= 1 for score in per_question_scores):
        raise ValueError("Provide at least one score, with each score between zero and one.")
    data = load_stats()
    t = data["topics"].get(topic) or _empty_topic()

    # Track raw history (cap to last 50 to keep file small)
    t["history"].extend(per_question_scores)
    t["history"] = t["history"][-50:]
    t["attempts"] += len(per_question_scores)
    t["correct"] += sum(per_question_scores)

    # Adapt difficulty based on rolling avg over recent history
    recent = t["history"][-10:]
    avg = sum(recent) / len(recent) if recent else 0.0
    if len(recent) >= 3 and avg > 0.70:
        t["difficulty"] = _bump(t["difficulty"], +1)
    elif len(recent) >= 3 and avg < 0.40:
        t["difficulty"] = _bump(t["difficulty"], -1)
    # else stay

    data["topics"][topic] = t
    save_stats(data)
    return t


def get_difficulty(topic: str) -> str:
    data = load_stats()
    t = data["topics"].get(topic)
    return t["difficulty"] if t else DEFAULT_DIFFICULTY


def format_stats_report() -> str:
    """Return a printable per-topic report."""
    data = load_stats()
    topics = data.get("topics", {})
    if not topics:
        return "No stats yet. Run a quiz first."
    lines = ["Topic stats:"]
    for name, t in sorted(topics.items()):
        attempts = t.get("attempts", 0)
        if attempts == 0:
            avg_pct = 0.0
        else:
            avg_pct = 100.0 * t.get("correct", 0) / attempts
        lines.append(
            f"  - {name}: attempts={attempts}, avg={avg_pct:.1f}%, difficulty={t.get('difficulty', DEFAULT_DIFFICULTY)}"
        )
    return "\n".join(lines)
