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


def _question_list(raw: str) -> list[dict]:
    data = json.loads(_strip_fences(raw))
    if not isinstance(data, dict) or not isinstance(data.get("questions"), list):
        raise ValueError("Model output must contain a questions list.")
    return data["questions"]


def _nonempty_text(value) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _valid_mcq(question) -> bool:
    if not isinstance(question, dict):
        return False
    options = question.get("options")
    return (
        _nonempty_text(question.get("question"))
        and isinstance(options, dict)
        and set(options) == {"A", "B", "C", "D"}
        and all(_nonempty_text(option) for option in options.values())
        and len({option.strip().casefold() for option in options.values()}) == 4
        and isinstance(question.get("answer"), str)
        and question["answer"] in options
        and _nonempty_text(question.get("explanation"))
    )


def _validate_count(n: int) -> None:
    if type(n) is not int or not 1 <= n <= 20:
        raise ValueError("Request between 1 and 20 questions.")


def _take_valid(questions: list[dict], n: int) -> list[dict]:
    """Keep the model's order and remove duplicates without filling from rejects."""
    selected, seen = [], set()
    for question in questions:
        key = " ".join(question["question"].casefold().split())
        if key not in seen:
            selected.append(question)
            seen.add(key)
        if len(selected) == n:
            break
    if not selected:
        raise ValueError("No questions passed validation. Try generating the quiz again.")
    return selected


def generate_mcqs(index: NotesIndex, n: int = 5) -> list[dict]:
    _validate_count(n)
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

    if not passages:
        raise ValueError("No passages found in the notes.")
    joined = "\n\n---\n\n".join(f"PASSAGE {i+1}:\n{p}" for i, p in enumerate(passages))
    user_msg = (
        f"Generate exactly {n} multiple-choice questions grounded in the following passages.\n\n"
        f"{joined}\n\nReturn JSON only."
    )
    questions = _question_list(complete(QUIZ_SYSTEM, user_msg, max_tokens=3000))
    return _take_valid([q for q in questions if _valid_mcq(q)], n)


def run_quiz(questions: list[dict]) -> list[float]:
    """Return completed scores in answer order, including on cancellation."""
    scores = []
    for i, q in enumerate(questions, 1):
        print(f"\nQ{i}. {q['question']}")
        for letter in ("A", "B", "C", "D"):
            print(f"  {letter}) {q['options'][letter]}")
        while True:
            try:
                ans = input("Your answer (A/B/C/D): ").strip().upper()
            except (EOFError, KeyboardInterrupt):
                print("\nQuiz stopped. Keeping completed answers only.")
                return scores
            if ans in ("A", "B", "C", "D"):
                break
            print("  Please type A, B, C, or D.")
        correct = q["answer"]
        if ans == correct:
            print(f"✓ Correct. {q.get('explanation', '')}")
            scores.append(1.0)
        else:
            print(f"✗ Wrong. Correct answer: {correct}. {q.get('explanation', '')}")
            scores.append(0.0)
    return scores


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
    _validate_count(n)
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

    if not passages:
        raise ValueError("No passages found in the notes.")
    joined = "\n\n---\n\n".join(f"PASSAGE {i+1}:\n{p}" for i, p in enumerate(passages))
    user_msg = (
        f"Generate exactly {n} open-ended questions grounded in these passages. "
        f"For each, copy the answering passage verbatim into 'source_passage'.\n\n"
        f"{joined}\n\nReturn JSON only."
    )
    questions = _question_list(complete(OPEN_GEN_SYSTEM, user_msg, max_tokens=2000))
    sources = {passage.strip() for passage in passages}
    valid = [
        q for q in questions
        if isinstance(q, dict) and _nonempty_text(q.get("question"))
        and _nonempty_text(q.get("source_passage")) and q["source_passage"].strip() in sources
    ]
    return _take_valid(valid, n)


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
    if not isinstance(data, dict) or type(data.get("score")) is not int or not 0 <= data["score"] <= 10:
        raise ValueError("The grader must return an integer score from zero to ten.")
    if not _nonempty_text(data.get("feedback")):
        raise ValueError("The grader returned no feedback.")
    return {"score": data["score"], "feedback": data["feedback"]}


def run_open_quiz(questions: list[dict]) -> list[int]:
    """Return completed grades only; failed grading is not an incorrect answer."""
    scores = []
    for i, q in enumerate(questions, 1):
        print(f"\nQ{i}. {q['question']}")
        try:
            answer = input("Your answer: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return scores
        if not answer:
            print("(empty answer — scored 0)")
            print(f"Source passage:\n  {q['source_passage']}")
            scores.append(0)
            continue
        try:
            result = grade_open_answer(q["question"], q["source_passage"], answer)
        except Exception as exc:
            print(f"Could not grade this answer: {exc}. No score recorded.")
            continue
        scores.append(result["score"])
        print(f"Score: {result['score']}/10")
        print(f"Feedback: {result['feedback']}")
        print(f"Source passage:\n  {q['source_passage']}")
    return scores


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
    if question.get("claimed_difficulty") != requested:
        return False
    q_text = question.get("question", "").lower()
    # Heuristic: hard questions usually contain scenario/application cues.
    hard_cues = ("suppose", "imagine", "scenario", "would happen", "predict", "apply", "given that", "if a ")
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
    _validate_count(n)
    if not _nonempty_text(topic):
        raise ValueError("A non-empty topic is required.")
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
    questions = _question_list(complete(QUIZ_TOPIC_SYSTEM, user_msg, max_tokens=3000))
    filtered = [q for q in questions if _valid_mcq(q) and _is_difficulty_consistent(q, difficulty)]
    return _take_valid(filtered, n)
