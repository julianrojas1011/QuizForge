"""Run the real quiz and history code with a hand-authored fixture, entirely offline."""
import argparse
import json
import tempfile
from contextlib import nullcontext
from pathlib import Path
from unittest.mock import patch

from quiz import run_quiz
from stats import record_results


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline QuizForge example: no API key, dependencies, or downloads.")
    parser.add_argument("--interactive", action="store_true", help="Answer the fixture questions yourself.")
    args = parser.parse_args()
    fixture = json.loads((Path(__file__).parent / "examples" / "quiz.json").read_text(encoding="utf-8"))
    print("Offline example — hand-authored questions, not a live model response.")
    print("The quiz runner and history update use the same code as the live CLI.\n")
    answers = iter(["C", "A", "C"])

    def scripted_answer(prompt):
        answer = next(answers)
        print(prompt + answer)
        return answer

    interaction = nullcontext() if args.interactive else patch("builtins.input", side_effect=scripted_answer)
    with tempfile.TemporaryDirectory() as directory, patch("stats.STATS_PATH", Path(directory) / "performance.json"), interaction:
        scores = run_quiz(fixture["questions"])
        if scores:
            topic = record_results(fixture["topic"], scores)
            print(f"\nScore: {int(sum(scores))}/{len(scores)}")
            print(f"Answer order: {topic['history']}")
            print(f"Next difficulty: {topic['difficulty']}")
    print("\nNo API calls or model downloads. Your performance.json was not changed.")


if __name__ == "__main__":
    main()
