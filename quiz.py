"""MCQ generation grounded in retrieved note chunks."""
import json
import re
from llm import complete
from rag import NotesIndex

QUIZ_SYSTEM = """You are a study quiz generator.

You receive PASSAGES from a student's notes. Write multiple-choice questions that test understanding of those passages.

Hard rules:
- Every question MUST be answerable from the passages alone.
- Do NOT introduce facts, terms, or examples not present in the passages.
- Each question has exactly 4 options A/B/C/D, exactly one correct.
- Distractors must be plausible but clearly wrong based on the passages.
- Output ONLY valid JSON matching the schema. No prose, no markdown fences.

Schema:
{
  "questions": [
    {
      "question": "string",
      "options": {"A": "string", "B": "string", "C": "string", "D": "string"},
      "answer": "A" | "B" | "C" | "D",
      "explanation": "one short sentence citing the passage"
    }
  ]
}
"""


def _strip_fences(s: str) -> str:
    s = s.strip()
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?\s*", "", s)
        s = re.sub(r"\s*```$", "", s)
    return s.strip()


def generate_mcqs(index: NotesIndex, n: int = 5) -> list[dict]:
    seed_queries = [
        "core definitions and key concepts",
        "differences between approaches",
        "common pitfalls and risks",
        "metrics and evaluation",
        "practical examples and applications",
    ]
    seen = set()
    passages: list[str] = []
    for q in seed_queries:
        for chunk in index.search(q, k=2):
            if chunk not in seen:
                seen.add(chunk)
                passages.append(chunk)
        if len(passages) >= 6:
            break

    joined = "\n\n---\n\n".join(f"PASSAGE {i+1}:\n{p}" for i, p in enumerate(passages))
    user_msg = (
        f"Generate exactly {n} multiple-choice questions grounded in the following passages.\n\n"
        f"{joined}\n\nReturn JSON only."
    )
    raw = _strip_fences(complete(QUIZ_SYSTEM, user_msg, max_tokens=3000))
    data = json.loads(raw)
    qs = data["questions"][:n]
    if len(qs) < n:
        raise ValueError(f"LLM returned {len(qs)} questions, expected {n}")
    return qs


def run_quiz(questions: list[dict]) -> int:
    score = 0
    for i, q in enumerate(questions, 1):
        print(f"\nQ{i}. {q['question']}")
        for letter in ("A", "B", "C", "D"):
            print(f"  {letter}) {q['options'][letter]}")
        while True:
            ans = input("Your answer (A/B/C/D): ").strip().upper()
            if ans in ("A", "B", "C", "D"):
                break
            print("  Please type A, B, C, or D.")
        correct = q["answer"]
        if ans == correct:
            print(f"✓ Correct. {q.get('explanation', '')}")
            score += 1
        else:
            print(f"✗ Wrong. Correct answer: {correct}. {q.get('explanation', '')}")
    return score


# ----- Milestone 2: open-ended questions -----

OPEN_GEN_SYSTEM = """You are a study question generator.

You will be given PASSAGES from a student's notes. Generate open-ended questions that test conceptual understanding of those passages.

Hard rules:
- Each question MUST be answerable from a single passage.
- Questions should require 1-3 sentences to answer (definitions, comparisons, brief explanations).
- Do NOT introduce facts or terms not in the passages.
- Output ONLY valid JSON. No prose, no markdown fences.

Schema:
{
  "questions": [
    {"question": "string", "source_passage": "string (verbatim copy of the passage that answers it)"}
  ]
}
"""


OPEN_GRADE_SYSTEM = """You are a fair, rigorous grader of short written answers.

You will receive:
- QUESTION: the question asked
- SOURCE_PASSAGE: the authoritative passage from the student's notes
- STUDENT_ANSWER: what the student wrote

Your job: grade the STUDENT_ANSWER from 0 to 10 based on how well it captures the meaning of the SOURCE_PASSAGE.

CRITICAL GRADING PRINCIPLES (read carefully):
1. Grade MEANING, not wording. A correct answer expressed in completely different words from the passage is FULLY CORRECT and must receive a high score (8-10).
2. Synonyms, paraphrases, and rephrasings are equivalent to the original. For example, "labeled data" and "training data with known outputs" describe the same concept and must be treated as equivalent.
3. Do NOT require the student to use specific terminology from the passage. If the student conveys the right idea using everyday language or alternative technical terms, that is correct.
4. Only deduct points for: missing key concepts, factual errors, contradictions with the passage, or vagueness that fails to capture the core idea.
5. Partial credit is appropriate when the answer captures some but not all of the key idea.

Score guide:
- 9-10: captures the full meaning of the passage, even if worded entirely differently
- 7-8: captures the main idea with minor omissions or imprecision
- 4-6: partially correct; gets some of the idea but misses or distorts important parts
- 1-3: largely incorrect or off-topic, with at most a tangential connection
- 0: completely wrong, blank, or unrelated

Output ONLY valid JSON, no prose, no markdown fences:
{"score": <int 0-10>, "feedback": "<one short sentence: what was right, what was missing, or what was wrong>"}
"""


def generate_open_questions(index: NotesIndex, n: int = 3) -> list[dict]:
    """Generate n open-ended questions, each with its source passage."""
    seed_queries = [
        "core definitions",
        "trade-offs and comparisons",
        "evaluation and metrics",
        "common pitfalls",
        "practical examples",
    ]
    seen = set()
    passages: list[str] = []
    for q in seed_queries:
        for chunk in index.search(q, k=2):
            if chunk not in seen:
                seen.add(chunk)
                passages.append(chunk)
        if len(passages) >= 5:
            break

    joined = "\n\n---\n\n".join(f"PASSAGE {i+1}:\n{p}" for i, p in enumerate(passages))
    user_msg = (
        f"Generate exactly {n} open-ended questions grounded in these passages. "
        f"For each, copy the answering passage verbatim into 'source_passage'.\n\n"
        f"{joined}\n\nReturn JSON only."
    )
    raw = _strip_fences(complete(OPEN_GEN_SYSTEM, user_msg, max_tokens=2000))
    data = json.loads(raw)
    qs = data["questions"][:n]
    if len(qs) < n:
        raise ValueError(f"LLM returned {len(qs)} questions, expected {n}")
    return qs


def grade_open_answer(question: str, source_passage: str, student_answer: str) -> dict:
    """Grade a single open-ended answer. Returns {'score': int, 'feedback': str}."""
    user_msg = (
        f"QUESTION:\n{question}\n\n"
        f"SOURCE_PASSAGE:\n{source_passage}\n\n"
        f"STUDENT_ANSWER:\n{student_answer}\n\n"
        f"Return JSON only."
    )
    raw = _strip_fences(complete(OPEN_GRADE_SYSTEM, user_msg, max_tokens=400))
    data = json.loads(raw)
    score = int(data["score"])
    score = max(0, min(10, score))
    return {"score": score, "feedback": data.get("feedback", "")}


def run_open_quiz(questions: list[dict]) -> int:
    """Ask each open question, grade it, print feedback + source. Return total score."""
    total = 0
    for i, q in enumerate(questions, 1):
        print(f"\nQ{i}. {q['question']}")
        try:
            answer = input("Your answer: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return total
        if not answer:
            print("(empty answer — scored 0)")
            print(f"Source passage:\n  {q['source_passage']}")
            continue
        result = grade_open_answer(q["question"], q["source_passage"], answer)
        total += result["score"]
        print(f"Score: {result['score']}/10")
        print(f"Feedback: {result['feedback']}")
        print(f"Source passage:\n  {q['source_passage']}")
    return total


# ----- Milestone 3: difficulty-aware MCQs scoped to a topic -----

DIFFICULTY_INSTRUCTIONS = {
    "easy": "Easy — test recall and definitions. Ask what something is, who did what, basic facts directly stated in the passage.",
    "medium": "Medium — test understanding and comparison. Ask the student to distinguish between concepts, identify why something works, or compare two ideas from the passage.",
    "hard": "Hard — test application and analysis. Give a short scenario or example and ask the student to apply the concept, predict an outcome, or reason about a non-obvious implication grounded in the passage.",
}


QUIZ_TOPIC_SYSTEM = """You are a study quiz generator that produces multiple-choice questions at a specific difficulty level on a specific topic.

You will be given:
- TOPIC: the topic name to quiz on
- DIFFICULTY: easy, medium, or hard
- DIFFICULTY_GUIDE: instructions for what that level means
- PASSAGES: passages from the student's notes about the topic

Hard rules:
- Every question MUST be answerable from the passages alone.
- Do NOT introduce facts, terms, or examples not in the passages.
- Each question MUST match the requested DIFFICULTY level. Re-read DIFFICULTY_GUIDE before each question.
- Each question has exactly 4 options A/B/C/D, exactly one correct.
- Output ONLY valid JSON. No prose, no markdown fences.

Schema:
{
  "questions": [
    {
      "question": "string",
      "options": {"A": "string", "B": "string", "C": "string", "D": "string"},
      "answer": "A" | "B" | "C" | "D",
      "explanation": "one short sentence",
      "claimed_difficulty": "easy" | "medium" | "hard"
    }
  ]
}

The "claimed_difficulty" field is REQUIRED. Set it to the difficulty you actually targeted for that question.
"""


def _is_difficulty_consistent(question: dict, requested: str) -> bool:
    """Cheap heuristic check + claimed_difficulty match."""
    claimed = (question.get("claimed_difficulty") or "").strip().lower()
    if claimed and claimed != requested:
        return False
    q_text = question.get("question", "").lower()
    # Heuristic: hard questions usually contain scenario/application cues.
    hard_cues = ("suppose", "imagine", "scenario", "would happen", "predict", "apply", "given that", "if a ")
    easy_cues = ("what is", "which of the following is the definition", "what does", "who")
    if requested == "hard":
        # Reject obviously-easy phrasings flagged as hard.
        if any(q_text.startswith(c) for c in ("what is ", "what does ")) and "?" in q_text and len(q_text) < 80:
            return False
    if requested == "easy":
        if any(c in q_text for c in hard_cues):
            return False
    return True


def generate_mcqs_for_topic(
    index: NotesIndex,
    topic: str,
    difficulty: str,
    n: int = 5,
) -> list[dict]:
    """Retrieve passages relevant to topic, generate n MCQs at the given difficulty."""
    if difficulty not in DIFFICULTY_INSTRUCTIONS:
        raise ValueError(f"Unknown difficulty: {difficulty}")

    # Retrieve chunks specifically about this topic.
    seen = set()
    passages: list[str] = []
    for query in (topic, f"{topic} examples", f"{topic} definition"):
        for chunk in index.search(query, k=3):
            if chunk not in seen:
                seen.add(chunk)
                passages.append(chunk)
        if len(passages) >= 5:
            break
    if not passages:
        raise ValueError(f"No passages found for topic: {topic}")

    joined = "\n\n---\n\n".join(f"PASSAGE {i+1}:\n{p}" for i, p in enumerate(passages))
    user_msg = (
        f"TOPIC: {topic}\n"
        f"DIFFICULTY: {difficulty}\n"
        f"DIFFICULTY_GUIDE: {DIFFICULTY_INSTRUCTIONS[difficulty]}\n\n"
        f"PASSAGES:\n{joined}\n\n"
        f"Generate exactly {n} MCQs at the {difficulty} level. Return JSON only."
    )
    raw = _strip_fences(complete(QUIZ_TOPIC_SYSTEM, user_msg, max_tokens=3000))
    data = json.loads(raw)
    qs = data["questions"][:n]

    # Filter out questions that fail the consistency check, then top up if short.
    filtered = [q for q in qs if _is_difficulty_consistent(q, difficulty)]
    # If too many were filtered, keep what we have plus the rejected ones to reach n.
    if len(filtered) < n:
        for q in qs:
            if q not in filtered and len(filtered) < n:
                filtered.append(q)
    return filtered[:n]
