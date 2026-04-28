# QuizForge

Terminal-based study tool. Quizzes you on a notes file, grades open-ended answers, adapts difficulty to your performance, and generates a personalized study plan via tool-calling.

## Setup

    python3 -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt
    cp .env.example .env  # then add your ANTHROPIC_API_KEY

## Usage

    python main.py --notes ml_basics.txt

REPL commands: /quiz, /quiz open, /stats, /plan, /exit.
