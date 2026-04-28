"""Per-topic performance tracking with persistence to performance.json."""
import json
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
        if "topics" not in data or not isinstance(data["topics"], dict):
            return {"topics": {}}
        return data
    except (json.JSONDecodeError, OSError):
        return {"topics": {}}


def save_stats(data: dict) -> None:
    with STATS_PATH.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


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
    data = load_stats()
    t = data["topics"].get(topic) or _empty_topic()

    # Track raw history (cap to last 50 to keep file small)
    t["history"].extend(per_question_scores)
    t["history"] = t["history"][-50:]
    t["attempts"] += len(per_question_scores)
    t["correct"] += sum(per_question_scores)

    # Adapt difficulty based on rolling avg over recent history
    recent = t["history"][-10:] if len(t["history"]) >= 3 else t["history"]
    avg = sum(recent) / len(recent) if recent else 0.0
    if avg > 0.70:
        t["difficulty"] = _bump(t["difficulty"], +1)
    elif avg < 0.40:
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
