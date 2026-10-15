import io
import os
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import llm
import main
import stats

ROOT = Path(__file__).resolve().parents[1]


class CliTests(unittest.TestCase):
    def test_help_runs_without_api_key_or_loading_embedding_library(self):
        env = dict(os.environ)
        env.pop("ANTHROPIC_API_KEY", None)
        result = subprocess.run([sys.executable, "main.py", "--help"], cwd=ROOT, env=env, capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--notes", result.stdout)

    def test_offline_demo_matches_committed_transcript(self):
        env = dict(os.environ)
        env.pop("ANTHROPIC_API_KEY", None)
        result = subprocess.run([sys.executable, "demo.py"], cwd=ROOT, env=env, capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, (ROOT / "examples" / "session.txt").read_text())

    def test_missing_key_has_actionable_error(self):
        with patch.dict(os.environ, {}, clear=True), patch.dict(sys.modules, {"dotenv": SimpleNamespace(load_dotenv=lambda: None)}):
            with self.assertRaisesRegex(ValueError, "Set ANTHROPIC_API_KEY"):
                llm.configure()

    def test_model_is_configurable(self):
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-only", "ANTHROPIC_MODEL": "test-model"}), patch.dict(sys.modules, {"dotenv": SimpleNamespace(load_dotenv=lambda: None)}):
            self.assertEqual(llm.configure(), "test-model")

    def test_completion_rejects_truncated_output(self):
        client = Mock()
        client.messages.create.return_value = SimpleNamespace(stop_reason="max_tokens", content=[])
        with patch("llm.get_client", return_value=client), patch("llm.configure", return_value="test"), self.assertRaisesRegex(ValueError, "incomplete"):
            llm.complete("System", "User")

    def test_cli_records_true_sequence_from_quiz(self):
        with tempfile.TemporaryDirectory() as directory, patch("stats.STATS_PATH", Path(directory) / "performance.json"), patch("main.generate_mcqs_for_topic", return_value=[{}] * 4), patch("main.run_quiz", return_value=[0.0, 1.0, 0.0, 1.0]), patch("builtins.input", return_value="1"), redirect_stdout(io.StringIO()):
            main.handle_command("/quiz", None, "notes", ["Evaluation"])
            self.assertEqual(stats.load_stats()["topics"]["Evaluation"]["history"], [0.0, 1.0, 0.0, 1.0])

    def test_cancel_without_answers_leaves_history_unchanged(self):
        with patch("main.get_difficulty", return_value="easy"), patch("main.generate_mcqs_for_topic", return_value=[{}]), patch("main.run_quiz", return_value=[]), patch("main.record_results") as record, patch("builtins.input", return_value="1"), redirect_stdout(io.StringIO()):
            main.handle_command("/quiz", None, "notes", ["Evaluation"])
            record.assert_not_called()


if __name__ == "__main__":
    unittest.main()
