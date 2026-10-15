# QuizForge

A personal project I built to learn retrieval and tool-calling through a study tool. It turns a text file of notes into quizzes, tracks multiple-choice results by topic, and uses that record to suggest what to study next.

## What it does

- Retrieves relevant passages with local sentence embeddings.
- Generates multiple-choice and short written-answer questions.
- Uses an LLM to grade written answers against a source passage and explain the score.
- Adjusts multiple-choice difficulty using recent results.
- Gives a study planner tools to read the notes outline and performance history.

## Try it

You need Python, a virtual environment, and an Anthropic API key. The first run downloads the local embedding model; question generation and grading use the API.

```bash
git clone https://github.com/julianrojas1011/QuizForge.git
cd QuizForge
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

On Windows, activate with `.venv\Scripts\activate` instead. Add `ANTHROPIC_API_KEY` to `.env`. The model identifier is set in `llm.py`; it must be available to your account.

```bash
python main.py --notes ml_basics.txt
```

| Command | What happens |
| --- | --- |
| `/quiz` | Choose a topic and take a multiple-choice quiz. Results update that topic's difficulty. |
| `/quiz open` | Answer written questions and see a model-generated score, feedback, and source passage. |
| `/stats` | View saved multiple-choice performance by topic. |
| `/plan` | Ask the tool-calling planner for a study plan. |
| `/topics` | List topics and their current difficulty. |
| `/help`, `/exit` | Show commands or finish. |

## How it works

`rag.py` builds the notes index. `quiz.py` handles question generation and grading. `stats.py` saves results to `performance.json`. `planner.py` runs a tool-calling loop with access to the topic outline, performance summary, and weak topics.

Use numbered headings in your notes, such as `1. Linear regression`, so the topic extractor can find them.

## Current limits

This is an experiment, not a validated assessment tool. Written-answer scores depend on the model and do not currently update the saved topic statistics. Difficulty checks are heuristic; rejected questions can be reintroduced to fill a quiz. Planner topic restrictions are prompt-based and are not checked against the outline after generation.

Retrieved passages and answers are sent to Anthropic. Embeddings run locally; quiz history is stored locally in `performance.json`.

[Implementation notes](DECISIONS.md) cover the choices and remaining limitations.
