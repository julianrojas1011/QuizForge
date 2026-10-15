import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import stats


class StatsTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "performance.json"
        self.patcher = patch("stats.STATS_PATH", self.path)
        self.patcher.start()
        self.addCleanup(self.patcher.stop)

    def test_missing_file_starts_empty(self):
        self.assertEqual(stats.load_stats(), {"topics": {}})

    def test_order_is_preserved_across_sessions_and_history_cap(self):
        first = [1.0, 0.0] * 24
        second = [0.0, 1.0, 1.0, 0.0, 0.0]
        stats.record_results("Topic", first)
        result = stats.record_results("Topic", second)
        self.assertEqual(result["history"], (first + second)[-50:])
        self.assertEqual(result["attempts"], 53)
        self.assertEqual(result["correct"], sum(first + second))

    def test_difficulty_uses_actual_last_ten_results(self):
        stats.record_results("Topic", [0.0] * 5)
        result = stats.record_results("Topic", [0.0] * 5 + [1.0] * 10)
        self.assertEqual(result["difficulty"], "medium")

    def test_small_sample_does_not_change_difficulty(self):
        self.assertEqual(stats.record_results("Topic", [1.0, 1.0])["difficulty"], "easy")

    def test_difficulty_clamps_and_exact_thresholds_do_not_bump(self):
        for _ in range(4):
            result = stats.record_results("Topic", [1.0] * 10)
        self.assertEqual(result["difficulty"], "hard")
        self.assertEqual(stats.record_results("Topic", [1.0] * 7 + [0.0] * 3)["difficulty"], "hard")
        self.assertEqual(stats.record_results("Topic", [1.0] * 4 + [0.0] * 6)["difficulty"], "hard")
        for _ in range(4):
            result = stats.record_results("Topic", [0.0] * 10)
        self.assertEqual(result["difficulty"], "easy")

    def test_invalid_results_do_not_write(self):
        for scores in [[], [True], [float("nan")], [float("inf")], [1.1], [-0.1], ["1"]]:
            with self.subTest(scores=scores), self.assertRaises(ValueError):
                stats.record_results("Topic", scores)
        self.assertFalse(self.path.exists())

    def test_malformed_stats_fail_without_overwriting(self):
        samples = ["not json", "null", "[]", '{"topics": []}', '{"topics":{"Math":{"difficulty":"unknown"}}}']
        for sample in samples:
            self.path.write_text(sample)
            with self.subTest(sample=sample), self.assertRaisesRegex(ValueError, "not been overwritten"):
                stats.record_results("Topic", [1.0])
            self.assertEqual(self.path.read_text(), sample)

    def test_saved_nonfinite_and_inconsistent_stats_are_rejected(self):
        valid = {"attempts": 1, "correct": 1.0, "difficulty": "easy", "history": [1.0]}
        for field, value in [("attempts", -1), ("correct", float("nan")), ("correct", 0), ("history", [2.0])]:
            sample = dict(valid, **{field: value})
            self.path.write_text(json.dumps({"topics": {"Math": sample}}))
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                stats.load_stats()

    def test_atomic_write_failure_keeps_previous_file(self):
        stats.record_results("Topic", [1.0])
        original = self.path.read_text()
        with patch("stats.os.replace", side_effect=OSError("disk error")), self.assertRaises(OSError):
            stats.record_results("Topic", [0.0])
        self.assertEqual(self.path.read_text(), original)
        self.assertEqual(list(self.path.parent.iterdir()), [self.path])


if __name__ == "__main__":
    unittest.main()
