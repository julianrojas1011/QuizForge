# Implementation notes

## Retrieval

`rag.py` packs paragraphs into chunks of up to 180 words with 30 words of overlap, splitting long paragraphs when needed. `all-MiniLM-L6-v2` embeds the chunks locally; retrieval ranks normalized vectors by dot product. Word counts do not guarantee token counts fit the embedding model, particularly for unusually long words or other languages.

The chunk sizes are starting parameters, not values established by a retrieval benchmark. Numbered, underlined section headings distinguish topics from numbered steps inside a section. The bundled notes provide ten such topics.

## Model output is a boundary

`quiz.py` accepts multiple-choice questions only when they have nonempty text, four distinct options, a valid answer key, and an explanation. Topic quizzes also require the requested difficulty label and pass a small wording heuristic. Duplicate or rejected questions never reenter a quiz to fill its requested count; a shorter quiz is reported, and an empty valid result stops the command.

For written questions, the returned source passage must match one of the retrieved passages. Grades must be integer scores from 0 to 10 with feedback. Invalid responses and incomplete model output are rejected instead of converted into plausible scores. Failed grading is excluded from the score denominator.

These checks verify structure and provenance, not semantic correctness. They do not prove that the answer key follows from the passage, that the question is useful, or that a grade is fair.

## Difficulty and history

Multiple-choice quizzes start at easy. `stats.py` moves difficulty up one step when the last ten scores average above 70%, down one step below 40%, and otherwise leaves it unchanged. At least three recorded answers are required before a change. These are hand-written rules, not a learned student model.

The CLI records outcomes in the order answered, including completed answers before a cancellation. It keeps the latest 50 scores and lifetime totals. Empty quizzes do not change history. Writes use a temporary file and atomic replacement; invalid saved data is reported without being overwritten.

History is keyed by topic name in the current working directory, so two note sets with the same topic share a record unless run in separate directories. This is a single-user CLI; simultaneous writers are not supported. Written-answer results are displayed but do not update this tracker.

## Study planning

`planner.py` exposes three read-only tools: `get_notes_outline`, `get_performance_summary`, and `get_weak_topics`. The loop allows up to six model responses, validates tool arguments, and returns tool errors in the tool-result protocol so the model can recover. A final answer is accepted only after successful outline and performance lookups.

The prompt requests canonical topic names and caution when there are few attempts. The final free-text plan is not validated against those topics. A future version could use structured plans and enforce topic membership before rendering them.

## What the tests establish

The offline suite exercises the data and control-flow contracts using fixed questions and mocked responses. `demo.py` uses hand-authored questions to demonstrate scoring and persistence without claiming that a model produced them. Neither is a live-model benchmark.

The next evaluation I would add is a small, versioned set of source passages, questions, and correct, incorrect, and paraphrased answers. I would compare retrieval relevance and grading behavior across changes, record failures, and keep that separate from any claim about improved learning outcomes.
