# QuizForge

A personal experiment I built to understand retrieval, model-generated questions, and tool-calling workflows through a study CLI. I used it to explore how these pieces fit together before treating them as something other people should rely on.

## Try the offline example

```bash
git clone https://github.com/julianrojas1011/QuizForge.git
cd QuizForge
python3 demo.py
```

This runs the real quiz and history code with **hand-authored questions and scripted answers**. No API key, packages, or model downloads are needed. It does not demonstrate live retrieval or generation, and it leaves your saved history unchanged.

```text
Score: 2/3
Answer order: [1.0, 0.0, 1.0]
Next difficulty: easy
```

[See the full session](examples/session.txt), or run `python3 demo.py --interactive` to answer the example questions yourself.

## What I explored

- **Retrieval:** chunk notes, embed them locally, and rank passages by similarity.
- **Generation and grading:** ask a model for questions and source-based feedback, then validate its structured output before displaying it.
- **State:** preserve answer order and use recent results to adjust the next quiz's difficulty with simple rules.
- **Tool orchestration:** let a study planner call read-only tools for the notes outline, performance history, and weaker topics.

This project uses existing models; it does not train a new model. Its difficulty rules and written-answer scores are experiments, not validated measures of learning.

## Run with your notes

Use **Python 3.11 or 3.12** and an Anthropic API key. The first live run downloads `all-MiniLM-L6-v2` for local embeddings. Question generation, written-answer grading, and planning use the API and may incur charges.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
cp .env.example .env
```

On Windows, activate with `.venv\Scripts\activate` instead. Set `ANTHROPIC_API_KEY` and, if needed, `ANTHROPIC_MODEL` in `.env` to a model available to your account.

```bash
python main.py --notes ml_basics.txt
```

| Command | What happens |
| --- | --- |
| `/quiz` | Pick a topic and answer up to five validated multiple-choice questions. Results update its history and difficulty. |
| `/quiz open` | Answer written questions and see a model-generated score, feedback, and the retrieved source passage. |
| `/plan` | Ask the tool-calling planner for a study plan using your outline and history. |
| `/stats`, `/topics` | Inspect saved multiple-choice performance or available topics. |
| `/help`, `/exit` | Show commands or finish. |

For reliable topic extraction, use numbered headings with an underline, as in `ml_basics.txt`:

```text
1. Supervised Learning
----------------------
Your notes go here.
```

## Verification

```bash
python -m unittest discover -s tests -v
```

The deterministic suite covers question rejection, source validation, score ordering, difficulty boundaries, damaged history files, chunking, and the bounded planner tool loop. It uses fixtures and mocked model responses: passing tests does **not** establish retrieval relevance or grading accuracy. GitHub Actions runs the suite on Python 3.11 and 3.12 and checks that the runtime dependencies install on Python 3.12.

## Scope and data

Written-answer grades do not update saved topic statistics. A source passage must match retrieved notes, but a question can still misinterpret that passage. Difficulty checks are heuristic, and the planner's final prose is not checked for unsupported topic names.

Retrieved passages, submitted answers, and planner tool results are sent to Anthropic. Embeddings and multiple-choice history stay local. History is stored in `performance.json` in the current directory, keyed by topic name; separate working directories keep unrelated note sets apart. If history is damaged, the CLI reports it without replacing the file—back it up, then repair or rename it.

[Implementation notes](DECISIONS.md) explain the tradeoffs and what I would evaluate next.
