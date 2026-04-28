"""QuizForge — terminal study tool. Entry point."""
import argparse
import sys
from pathlib import Path

from rag import NotesIndex, extract_topic_outline
from quiz import generate_mcqs, run_quiz, generate_open_questions, run_open_quiz, generate_mcqs_for_topic
from stats import record_results, get_difficulty, format_stats_report
from planner import generate_study_plan


def main() -> int:
    parser = argparse.ArgumentParser(description="QuizForge — quiz yourself from notes.")
    parser.add_argument("--notes", required=True, help="Path to a .txt notes file.")
    args = parser.parse_args()

    notes_path = Path(args.notes)
    if not notes_path.exists():
        print(f"Error: notes file not found: {notes_path}", file=sys.stderr)
        return 1

    print(f"Loading notes from {notes_path}...")
    text = notes_path.read_text(encoding="utf-8")
    print("Building index (embedding chunks)...")
    index = NotesIndex.build(text)
    topics = extract_topic_outline(text)
    if not topics:
        print("Warning: could not extract topic outline. Falling back to single 'General' topic.")
        topics = ["General"]
    print(f"Indexed {len(index.chunks)} chunks across {len(topics)} topics. Type /quiz, /quiz open, /plan, /stats, /topics, /help, /exit.")

    while True:
        try:
            cmd = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not cmd:
            continue
        if cmd in ("/exit", "/quit"):
            return 0
        if cmd == "/help":
            print("Commands: /quiz  /quiz open  /plan  /stats  /topics  /help  /exit")
            continue
        if cmd == "/plan":
            print("Generating personalized study plan (this may take ~10-20 seconds)...")
            try:
                plan = generate_study_plan(text)
            except Exception as e:
                print(f"Error generating plan: {e}")
                continue
            print()
            print(plan)
            continue
        if cmd == "/quiz open":
            try:
                qs = generate_open_questions(index, n=3)
            except Exception as e:
                print(f"Error generating open quiz: {e}")
                continue
            total = run_open_quiz(qs)
            print(f"\nTotal: {total}/{len(qs) * 10}")
            continue
        if cmd == "/topics":
            for i, t in enumerate(topics, 1):
                diff = get_difficulty(t)
                print(f"  {i}. {t}  [{diff}]")
            continue
        if cmd == "/stats":
            print(format_stats_report())
            continue
        if cmd == "/quiz":
            print("Pick a topic:")
            for i, t in enumerate(topics, 1):
                diff = get_difficulty(t)
                print(f"  {i}. {t}  [{diff}]")
            try:
                choice = input("Topic number: ").strip()
                idx = int(choice) - 1
                if not (0 <= idx < len(topics)):
                    print("Invalid topic number.")
                    continue
            except (ValueError, EOFError, KeyboardInterrupt):
                print("\nCancelled.")
                continue
            topic = topics[idx]
            difficulty = get_difficulty(topic)
            print(f"Generating {difficulty} quiz on '{topic}'...")
            try:
                qs = generate_mcqs_for_topic(index, topic, difficulty, n=5)
            except Exception as e:
                print(f"Error generating quiz: {e}")
                continue
            score = run_quiz(qs)
            print(f"\nYou scored {score}/{len(qs)}")
            per_q = [1.0] * score + [0.0] * (len(qs) - score)
            updated = record_results(topic, per_q)
            print(f"Topic difficulty is now: {updated['difficulty']}")
            continue
        print(f"Unknown command: {cmd}. Type /help.")


if __name__ == "__main__":
    sys.exit(main())
