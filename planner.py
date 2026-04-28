"""Study plan generator using Anthropic tool calling."""
import json
from anthropic import Anthropic
from dotenv import load_dotenv

from llm import MODEL
from rag import extract_topic_outline
from stats import load_stats, DIFFICULTIES

load_dotenv()
_client = Anthropic()


# --- Tool definitions exposed to the LLM ---

TOOL_DEFS = [
    {
        "name": "get_performance_summary",
        "description": (
            "Returns the full per-topic performance record for the student: for each topic the student has "
            "attempted, returns attempts, correct count, average accuracy, and current difficulty level. "
            "Returns an empty object if the student has not taken any quizzes yet."
        ),
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "get_weak_topics",
        "description": (
            "Returns topics whose average accuracy is BELOW the given threshold. Each entry includes attempts, "
            "average accuracy, and current difficulty. Use this to focus the plan on areas that need work. "
            "Note: a topic with very few attempts (e.g. 1-2) may appear weak by chance — the response includes "
            "the attempts count so you can judge confidence."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "threshold": {
                    "type": "number",
                    "description": "Accuracy threshold in [0.0, 1.0]. Topics with avg < threshold are returned.",
                }
            },
            "required": ["threshold"],
        },
    },
    {
        "name": "get_notes_outline",
        "description": (
            "Returns the canonical list of topic titles extracted from the student's notes file. "
            "These are the ONLY topics that exist in the study material. Any plan you generate must "
            "use topic names from this list — do not invent topics that are not in the outline."
        ),
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
]


# --- Tool implementations ---

def _tool_get_performance_summary() -> dict:
    data = load_stats()
    topics = data.get("topics", {})
    out = {}
    for name, t in topics.items():
        attempts = t.get("attempts", 0)
        correct = t.get("correct", 0)
        avg = (correct / attempts) if attempts else 0.0
        out[name] = {
            "attempts": attempts,
            "correct": round(correct, 2),
            "average_accuracy": round(avg, 3),
            "difficulty": t.get("difficulty", "easy"),
        }
    return out


def _tool_get_weak_topics(threshold: float) -> dict:
    threshold = float(threshold)
    summary = _tool_get_performance_summary()
    weak = {
        name: info
        for name, info in summary.items()
        if info["attempts"] > 0 and info["average_accuracy"] < threshold
    }
    return weak


def _tool_get_notes_outline(notes_text: str) -> list[str]:
    return extract_topic_outline(notes_text)


def _dispatch_tool(name: str, args: dict, notes_text: str):
    if name == "get_performance_summary":
        return _tool_get_performance_summary()
    if name == "get_weak_topics":
        return _tool_get_weak_topics(args.get("threshold", 0.5))
    if name == "get_notes_outline":
        return _tool_get_notes_outline(notes_text)
    raise ValueError(f"Unknown tool: {name}")


PLANNER_SYSTEM = """You are a personalized study plan generator.

You have access to three tools. You MUST use them to gather data — do NOT guess at the student's performance or the topics in their notes.

Required workflow:
1. First call `get_notes_outline` to learn which topics actually exist in the student's material.
2. Then call `get_performance_summary` (or `get_weak_topics` with a threshold like 0.5) to see how the student is doing.
3. Based on the data returned, produce a study plan.

CRITICAL RULES for the final plan:
- Every topic you recommend MUST appear verbatim in the outline returned by `get_notes_outline`. Never invent topics.
- For each topic, suggest a number of questions (3-10) and a starting difficulty (easy / medium / hard).
- Prioritize topics where the student is weakest; if a topic has very few attempts (under 3), explicitly note that the assessment is low-confidence and recommend a short diagnostic round before deeper work.
- If the student has no performance data at all, recommend an even baseline across all topics from the outline at easy difficulty.

After tool calls, output the final plan as a human-readable text response (NOT JSON). Use this structure:

```
STUDY PLAN
==========

Priority order:
  1. <topic>  —  <N> questions, starting difficulty: <easy|medium|hard>
     Reason: <one short sentence>
  2. ...

Notes:
  - <any caveats about low-confidence assessments, sparse data, etc.>
```
"""


def generate_study_plan(notes_text: str, max_iterations: int = 6) -> str:
    """Run the tool-call loop and return the final plan text."""
    messages: list[dict] = [
        {
            "role": "user",
            "content": "Generate a personalized study plan for me based on my notes and my recent performance. Use the tools to gather what you need.",
        }
    ]

    for _ in range(max_iterations):
        resp = _client.messages.create(
            model=MODEL,
            max_tokens=2048,
            system=PLANNER_SYSTEM,
            tools=TOOL_DEFS,
            messages=messages,
        )

        # Append the assistant turn (must include all blocks for tool_use continuity).
        messages.append({"role": "assistant", "content": resp.content})

        if resp.stop_reason == "end_turn":
            # Final answer — concatenate any text blocks.
            text_parts = [b.text for b in resp.content if getattr(b, "type", None) == "text"]
            return "\n".join(text_parts).strip() or "(planner returned no text)"

        if resp.stop_reason == "tool_use":
            tool_results = []
            for block in resp.content:
                if getattr(block, "type", None) == "tool_use":
                    name = block.name
                    args = block.input or {}
                    try:
                        result = _dispatch_tool(name, args, notes_text)
                        result_str = json.dumps(result)
                    except Exception as e:
                        result_str = json.dumps({"error": str(e)})
                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": result_str,
                        }
                    )
            messages.append({"role": "user", "content": tool_results})
            continue

        # Unexpected stop reason — return whatever text we have.
        text_parts = [b.text for b in resp.content if getattr(b, "type", None) == "text"]
        return "\n".join(text_parts).strip() or f"(planner stopped: {resp.stop_reason})"

    return "(planner exceeded max iterations without producing a plan)"
