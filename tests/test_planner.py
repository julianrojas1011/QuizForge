import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import planner

NOTES = "1. Evaluation\n---\n\nUse held-out data."


def response(reason, blocks):
    return SimpleNamespace(stop_reason=reason, content=blocks)


def tool(name, args=None, identifier="tool1"):
    return SimpleNamespace(type="tool_use", name=name, input={} if args is None else args, id=identifier)


class PlannerTests(unittest.TestCase):
    def test_invalid_tool_arguments_are_rejected(self):
        for value in ["0.5", True, -0.1, 1.1, float("nan"), float("inf")]:
            with self.subTest(value=value), self.assertRaises(ValueError):
                planner._dispatch_tool("get_weak_topics", {"threshold": value}, NOTES)
        for name, args in [("get_weak_topics", {}), ("get_notes_outline", []), ("get_notes_outline", {"unknown": 1}), ("unknown", {})]:
            with self.subTest(name=name, args=args), self.assertRaises(ValueError):
                planner._dispatch_tool(name, args, NOTES)

    def run_responses(self, responses, max_iterations=6):
        client = Mock()
        client.messages.create.side_effect = responses
        with patch("planner.get_client", return_value=client), patch("planner.configure", return_value="test-model"), patch("planner.load_stats", return_value={"topics": {}}):
            result = planner.generate_study_plan(NOTES, max_iterations=max_iterations)
        return result, client.messages.create.call_args_list

    def test_plan_requires_successful_outline_and_performance_tools(self):
        result, calls = self.run_responses([
            response("end_turn", [SimpleNamespace(type="text", text="Ungrounded draft")]),
            response("tool_use", [tool("get_notes_outline"), tool("get_performance_summary", identifier="tool2")]),
            response("end_turn", [SimpleNamespace(type="text", text="Study Evaluation.")]),
        ])
        self.assertEqual(result, "Study Evaluation.")
        self.assertEqual(len(calls), 3)
        results = calls[-1].kwargs["messages"][4]["content"]
        self.assertEqual([r["tool_use_id"] for r in results], ["tool1", "tool2"])
        self.assertTrue(all(not r["is_error"] for r in results))

    def test_bad_tool_result_is_marked_error_and_model_can_recover(self):
        result, calls = self.run_responses([
            response("tool_use", [tool("get_weak_topics", {"threshold": "not a number"})]),
            response("tool_use", [tool("get_notes_outline"), tool("get_performance_summary", identifier="tool2")]),
            response("end_turn", [SimpleNamespace(type="text", text="Study Evaluation.")]),
        ])
        self.assertEqual(result, "Study Evaluation.")
        self.assertTrue(calls[-1].kwargs["messages"][2]["content"][0]["is_error"])

    def test_loop_has_a_hard_limit(self):
        with self.assertRaisesRegex(ValueError, "tool-call limit"):
            self.run_responses([response("end_turn", [SimpleNamespace(type="text", text="No tools")])] * 2, max_iterations=2)

    def test_truncated_output_is_not_presented_as_a_plan(self):
        with self.assertRaisesRegex(ValueError, "incomplete"):
            self.run_responses([response("max_tokens", [SimpleNamespace(type="text", text="Partial plan")])])

    def test_notes_without_headings_stop_before_an_api_call(self):
        with patch("planner.get_client") as client, self.assertRaisesRegex(ValueError, "numbered"):
            planner.generate_study_plan("Unstructured notes")
        client.assert_not_called()


if __name__ == "__main__":
    unittest.main()
