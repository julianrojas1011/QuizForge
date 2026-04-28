# DECISIONS.md

## Milestone 1 — Chunking strategy

I chunk the notes into overlapping windows of ~180 words with 30 words of overlap, splitting on paragraph boundaries first and packing paragraphs until the target word count. The 180-word target was chosen because the source notes (`ml_basics.txt`) are organized into numbered sections of ~150–300 words each, so this size keeps a self-contained idea in one chunk most of the time. The 30-word overlap prevents losing a definition that straddles a chunk boundary.

If chunks were too small (one sentence each), retrieval would surface fragments without surrounding context — the LLM would see "Recall measures: of all the actual positive items, how many did the model catch?" with no anchor to *what* recall is part of, and would hallucinate or write vague questions. If chunks were too large (the whole file), embedding similarity becomes meaningless because every query is "close" to the one giant vector, and retrieval cannot focus the LLM on a specific topic, leading to generic, unfocused questions.

## Milestone 2 — Grading paraphrased answers

The exam case: notes say *"Supervised learning uses labeled data to train models"*; student writes *"In supervised learning, the training data has known outputs that the model learns to predict."* These are the same idea worded differently. A naive grader that does keyword overlap or asks the LLM "does this match?" will mark it wrong because the literal phrase "labeled data" never appears.

My grading prompt (in `quiz.py`, `OPEN_GRADE_SYSTEM`) is:

```
You are a fair, rigorous grader of short written answers.

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
```

The specific instruction that handles this case is **CRITICAL GRADING PRINCIPLE #1 and #2**: "Grade MEANING, not wording" and "Synonyms, paraphrases, and rephrasings are equivalent to the original." Without those two principles, the model's default behavior would be to reward surface-level overlap with the source passage, penalizing students who restate ideas in their own words — which is exactly the opposite of what you want from a study tool. Principle #3 ("Do NOT require specific terminology") is a backstop: it explicitly forbids the model from docking points just because a textbook term is missing, which is the most common failure mode of LLM-as-judge graders.

If I removed those principles, the grader would default to surface-level matching: the student in the example case would likely score 4-6 instead of 9-10, because the literal phrase "labeled data" is missing even though the meaning is fully captured. I verified this works as intended with a paraphrase unit test that asserts score ≥ 8 on the canonical exam case.

## Milestone 3 — Difficulty levels and disobedience handling

The 3 levels are encoded in `quiz.py` as `DIFFICULTY_INSTRUCTIONS`, which is interpolated into the per-quiz prompt:

```
easy: Easy — test recall and definitions. Ask what something is, who did what, basic facts directly stated in the passage.
medium: Medium — test understanding and comparison. Ask the student to distinguish between concepts, identify why something works, or compare two ideas from the passage.
hard: Hard — test application and analysis. Give a short scenario or example and ask the student to apply the concept, predict an outcome, or reason about a non-obvious implication grounded in the passage.
```

The LLM occasionally ignores difficulty — most often by emitting a recall-style "What is X?" question when asked for hard. I handle this **in code**, not by tweaking the prompt: every generated question carries a required `claimed_difficulty` field, and `_is_difficulty_consistent()` rejects questions whose `claimed_difficulty` disagrees with the request OR whose phrasing matches obvious-mismatch heuristics (e.g. a "hard" question that starts with "What is" and is short, or an "easy" question containing scenario cues like "suppose"/"predict"). Rejected questions are dropped from the returned list. If filtering leaves us short, the function tops up from the unfiltered set so the user still gets `n` questions, but the filter at least prevents the worst mismatches from making it through.

## Milestone 4 — Tool-calling, hallucination, and sparse-data guards

### A. Hallucination — fabricated topics

The LLM could plausibly recommend topics that are not in `ml_basics.txt` ("Neural Network Architectures", "Reinforcement Learning"), since those are common in any ML curriculum. My current defense is **prompt-side**: the planner system prompt explicitly says "Every topic you recommend MUST appear verbatim in the outline returned by `get_notes_outline`", and the workflow forces the LLM to call that tool first. But prompt instructions are not a hard guarantee.

**The real fix in code, not prompt:** after the planner returns its plan text, post-process it against the canonical outline. Concretely, in `planner.py` add a `_validate_plan(plan_text, allowed_topics)` step that runs after the tool loop ends — it tokenizes each line of the plan, looks for the topic-name pattern in the priority list, and for any topic name found in the plan that does not match (case-insensitive, after stripping punctuation) something in `extract_topic_outline(notes_text)`, the line is flagged or removed. If hallucinated entries exist, return them appended to the plan output as `WARNING: dropped fabricated topics: [...]`. This makes hallucination detectable to the user and recoverable rather than silent.

### B. Sparse data — over-confident weak-topic detection

If the student has answered only 2 questions on a topic, `get_weak_topics(0.5)` will happily return it as "weak" with no awareness that 2 attempts is statistically meaningless. The tool's return value already includes the `attempts` count, and the system prompt instructs the LLM to flag low-confidence cases (under 3 attempts) and recommend a diagnostic round — but this is again a soft guard.

**The fix in code:** modify `_tool_get_weak_topics` to attach a `confidence` field per topic, computed from `attempts` (e.g. `"low"` if attempts < 3, `"medium"` if 3-9, `"high"` if ≥ 10). Optionally add a `min_attempts` argument to the tool that filters out topics below that threshold entirely, so the LLM cannot use sparse data even if it tries. The planner can then surface confidence-tagged recommendations explicitly: a topic with one bad attempt becomes "diagnostic round (low-confidence assessment)" rather than "high-priority weak area".
