import io
import json
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

import quiz


class Notes:
    def search(self, query, k=5):
        return ["Training uses labeled examples. Evaluation uses held-out data."]


def mcq(text="Which set is used for evaluation?", **changes):
    q = {
        "question": text,
        "options": {"A": "Held-out data", "B": "Training labels", "C": "No data", "D": "The same examples"},
        "answer": "A", "explanation": "The passage says to use held-out data.",
        "claimed_difficulty": "easy",
    }
    q.update(changes)
    return q


class QuestionTests(unittest.TestCase):
    def topic_questions(self, questions, difficulty="easy", n=5):
        with patch("quiz.complete", return_value=json.dumps({"questions": questions})):
            return quiz.generate_mcqs_for_topic(Notes(), "Evaluation", difficulty, n)

    def test_rejected_difficulty_never_reenters_quiz(self):
        accepted = mcq()
        rejected = mcq("Suppose a model memorizes its training set?", claimed_difficulty="hard")
        self.assertEqual(self.topic_questions([rejected, accepted]), [accepted])

    def test_heuristic_rejection_is_not_reinserted(self):
        with self.assertRaisesRegex(ValueError, "No questions"):
            self.topic_questions([mcq("What is a model?", claimed_difficulty="hard")], "hard")

    def test_missing_claim_is_rejected(self):
        q = mcq()
        del q["claimed_difficulty"]
        with self.assertRaises(ValueError):
            self.topic_questions([q])

    def test_invalid_options_answers_and_text_are_rejected(self):
        mutations = [None, [], mcq(options={"A": "Only one"}), mcq(answer="Z"), mcq(answer=[]),
                     mcq(question=""), mcq(options={k: "same" for k in "ABCD"}), mcq(explanation=None)]
        valid = mcq()
        self.assertEqual(self.topic_questions(mutations + [valid]), [valid])

    def test_duplicate_questions_removed(self):
        q = mcq()
        duplicate = mcq("  WHICH   SET IS USED FOR EVALUATION? ")
        self.assertEqual(self.topic_questions([q, duplicate]), [q])

    def test_valid_questions_after_rejected_candidates_are_used(self):
        valid = mcq()
        self.assertEqual(self.topic_questions([None, valid], n=1), [valid])

    def test_non_list_response_is_rejected(self):
        for response in ["null", "[]", '{"questions":null}', '{"questions":{}}', "not json"]:
            with self.subTest(response=response), patch("quiz.complete", return_value=response):
                with self.assertRaises(ValueError):
                    quiz.generate_mcqs_for_topic(Notes(), "Evaluation", "easy")

    def test_invalid_count_rejected_before_model_call(self):
        for n in [0, -1, True, 21]:
            with patch("quiz.complete") as complete, self.assertRaises(ValueError):
                quiz.generate_mcqs(Notes(), n=n)
            complete.assert_not_called()

    def test_open_question_source_must_be_retrieved_verbatim(self):
        source = Notes().search("")[0]
        valid = {"question": "How should a model be evaluated?", "source_passage": source}
        invented = {"question": "What is AI?", "source_passage": "Invented evidence"}
        with patch("quiz.complete", return_value=json.dumps({"questions": [invented, valid]})):
            self.assertEqual(quiz.generate_open_questions(Notes()), [valid])

    def test_bad_grades_are_rejected_not_coerced_or_clamped(self):
        for score in [True, "9", 8.5, -1, 11, None]:
            with self.subTest(score=score), patch("quiz.complete", return_value=json.dumps({"score": score, "feedback": "Feedback"})):
                with self.assertRaises(ValueError):
                    quiz.grade_open_answer("Q", "Source", "Answer")

    def test_valid_grade_keeps_feedback(self):
        with patch("quiz.complete", return_value='{"score":8,"feedback":"Missing the evaluation step."}'):
            self.assertEqual(quiz.grade_open_answer("Q", "Source", "Answer")["score"], 8)

    def test_scores_preserve_answer_order(self):
        with patch("builtins.input", side_effect=["B", "A", "D", "A"]), redirect_stdout(io.StringIO()):
            self.assertEqual(quiz.run_quiz([mcq()] * 4), [0.0, 1.0, 0.0, 1.0])

    def test_cancel_keeps_only_completed_answers(self):
        with patch("builtins.input", side_effect=["B", EOFError]), redirect_stdout(io.StringIO()):
            self.assertEqual(quiz.run_quiz([mcq()] * 3), [0.0])

    def test_invalid_answer_reprompts_without_recording_an_attempt(self):
        with patch("builtins.input", side_effect=["invalid", "a"]), redirect_stdout(io.StringIO()):
            self.assertEqual(quiz.run_quiz([mcq()]), [1.0])

    def test_failed_grading_does_not_become_zero(self):
        q = {"question": "Q", "source_passage": "S"}
        with patch("builtins.input", return_value="Answer"), patch("quiz.grade_open_answer", side_effect=ValueError("bad response")), redirect_stdout(io.StringIO()):
            self.assertEqual(quiz.run_open_quiz([q]), [])


if __name__ == "__main__":
    unittest.main()
