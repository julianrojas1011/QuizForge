import unittest
from pathlib import Path

from rag import chunk_text, extract_topic_outline


class NotesTests(unittest.TestCase):
    def test_long_paragraphs_are_bounded_and_cover_input(self):
        words = [f"word{i}" for i in range(701)]
        chunks = chunk_text(" ".join(words))
        self.assertTrue(all(len(chunk.split()) <= 180 for chunk in chunks))
        self.assertEqual(set(" ".join(chunks).split()), set(words))
        for first, second in zip(chunks, chunks[1:]):
            self.assertEqual(first.split()[-30:], second.split()[:30])

    def test_zero_overlap_does_not_copy_the_whole_previous_chunk(self):
        text = "one two three\n\nfour five six"
        self.assertEqual(chunk_text(text, target_words=4, overlap_words=0), ["one two three", "four five six"])

    def test_invalid_chunk_sizes_fail(self):
        for size, overlap in [(0, 0), (2, 2), (2, -1), (True, 0)]:
            with self.assertRaises(ValueError):
                chunk_text("notes", size, overlap)

    def test_empty_notes_make_no_chunks(self):
        self.assertEqual(chunk_text(" \n  \n"), [])

    def test_bundled_notes_have_ten_topics_not_numbered_process_steps(self):
        text = (Path(__file__).resolve().parents[1] / "ml_basics.txt").read_text()
        topics = extract_topic_outline(text)
        self.assertEqual(len(topics), 10)
        self.assertNotIn("Collect labeled training data", topics)

    def test_simple_headings_supported_without_underlines(self):
        self.assertEqual(extract_topic_outline("1. First topic\n\nNotes\n\n2. Second topic\n\nNotes"), ["First topic", "Second topic"])


if __name__ == "__main__":
    unittest.main()
