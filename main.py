"""QuizForge — terminal study tool. Entry point."""
import argparse
import sys
from pathlib import Path

from llm import configure
from rag import NotesIndex, extract_topic_outline
from quiz import run_quiz, generate_open_questions, run_open_quiz, generate_mcqs_for_topic
from stats import record_results, get_difficulty, format_stats_report
from planner import generate_study_plan

COMMANDS = "Commands: /quiz  /quiz open  /plan  /stats  /topics  /help  /exit"


def handle_command(cmd: str, index: NotesIndex, notes_text: str, topics: list[str]) -> bool:
    """Handle one command. Return false when the user wants to leave."""
    if cmd in ("/exit", "/quit"):
        return False
    if not cmd:
        return True
    if cmd == "/help":
        print(COMMANDS)
    elif cmd == "/plan":
        print("Generating a study plan...")
        print(generate_study_plan(notes_text))
    elif cmd == "/quiz open":
        questions = generate_open_questions(index, n=3)
        if len(questions) < 3:
            print(f"{len(questions)} questions passed validation; using those only.")
        scores = run_open_quiz(questions)
        if scores:
            print(f"\nTotal: {sum(scores)}/{len(scores) * 10} across {len(scores)} graded answers.")
        else:
            print("No answers were graded.")
    elif cmd == "/topics":
        for i, topic in enumerate(topics, 1):
            print(f"  {i}. {topic}  [{get_difficulty(topic)}]")
    elif cmd == "/stats":
        print(format_stats_report())
    elif cmd == "/quiz":
        print("Pick a topic:")
        for i, topic in enumerate(topics, 1):
            print(f"  {i}. {topic}  [{get_difficulty(topic)}]")
        try:
            idx = int(input("Topic number: ").strip()) - 1
        except ValueError:
            print("Invalid topic number.")
            return True
        if not 0 <= idx < len(topics):
            print("Invalid topic number.")
            return True
        topic = topics[idx]
        difficulty = get_difficulty(topic)
        print(f"Generating {difficulty} quiz on '{topic}'...")
        questions = generate_mcqs_for_topic(index, topic, difficulty, n=5)
        if len(questions) < 5:
            print(f"{len(questions)} questions passed validation; using those only.")
        scores = run_quiz(questions)
        if scores:
            updated = record_results(topic, scores)
            print(f"\nYou scored {int(sum(scores))}/{len(scores)}")
            print(f"Topic difficulty is now: {updated['difficulty']}")
        else:
            print("No completed answers; performance history is unchanged.")
    else:
        print(f"Unknown command: {cmd}. Type /help.")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="QuizForge — quiz yourself from notes.")
    parser.add_argument("--notes", required=True, help="Path to a .txt notes file.")
    args = parser.parse_args()

    notes_path = Path(args.notes)
    try:
        text = notes_path.read_text(encoding="utf-8")
        if not text.strip():
            raise ValueError("The notes file is empty.")
        configure()
        print(f"Loading notes from {notes_path}...")
        print("Building index (embedding chunks)...")
        index = NotesIndex.build(text)
    except (OSError, UnicodeError, ValueError, RuntimeError, ImportError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    topics = extract_topic_outline(text)
    if not topics:
        print("No numbered headings found. Quizzes use 'General'; /plan requires numbered topics.")
        topics = ["General"]
    print(f"Indexed {len(index.chunks)} chunks across {len(topics)} topics. {COMMANDS}")

    while True:
        try:
            cmd = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        try:
            if not handle_command(cmd, index, text, topics):
                return 0
        except (EOFError, KeyboardInterrupt):
            print("\nCancelled.")
        except Exception as exc:
            print(f"Could not complete that command: {exc}")


if __name__ == "__main__":
    sys.exit(main())
