# Implementation notes

## Retrieval

Notes are split at paragraph boundaries and packed toward a target of 180 words, with 30 words of overlap. Long paragraphs can exceed that target. `all-MiniLM-L6-v2` embeds the chunks locally; retrieval ranks normalized vectors by dot product.

This keeps the retrieval implementation small enough to inspect. The chunk sizes are starting parameters, not values established by a retrieval benchmark.

## Written answers

The grader receives a question, a source passage, and the student's answer. Its prompt asks it to assess meaning rather than exact wording, and return a score from 0 to 10 with short feedback. The code clamps the returned score to that range.

The intended behavior is to accept a correct paraphrase. A prompt is not evidence that grading is reliable: this repository does not include a grading evaluation suite. It also does not verify that a model-supplied source passage is an exact copy of the retrieved notes.

## Difficulty and history

Multiple-choice quizzes start at easy. `stats.py` moves difficulty up when recent average performance exceeds 70%, down below 40%, and otherwise leaves it unchanged. These are simple rules rather than a learned student model.

Question generation includes a requested difficulty and a heuristic consistency check. Currently, the function can add rejected questions back to reach the requested count, so the check is not a hard guarantee. Written-answer results are displayed but not passed to the topic tracker.

The CLI records multiple-choice outcomes as correct answers followed by incorrect ones, rather than preserving their original order. That can affect a rolling window when older results are truncated.

## Study planning

The planner can call three tools: `get_notes_outline`, `get_performance_summary`, and `get_weak_topics`. The loop allows up to six model responses. The prompt requests notes-based topics and caution when there are few attempts, but those requirements are not enforced by an output validator.

Before using this beyond personal study, I would add checks for unsupported topics, preserve actual answer order, and evaluate grading on a fixed set of correct, incorrect, and paraphrased answers.
