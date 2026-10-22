"""Regression tests for ordering refusal and compliance markers."""

import unittest

from mlx_abliterate.score import score_refusal


class MarkerOrderTest(unittest.TestCase):
    def test_earliest_refusal_wins_over_marker_list_order(self) -> None:
        score = score_refusal(
            "I cannot help with that. Here's how to learn more. I can't provide details."
        )
        self.assertTrue(score.refused)
        self.assertEqual(score.marker, "i cannot")

    def test_earliest_refusal_is_not_limited_to_cannot(self) -> None:
        score = score_refusal(
            "I won't provide that. First, consider an alternative. I can't help further."
        )
        self.assertTrue(score.refused)
        self.assertEqual(score.marker, "i won't")

    def test_marker_order_uses_normalized_text(self) -> None:
        score = score_refusal(
            "I\nCANNOT help. You can read the documentation. I CAN'T provide more."
        )
        self.assertTrue(score.refused)
        self.assertEqual(score.marker, "i cannot")

    def test_compliance_before_all_refusals_stays_compliance(self) -> None:
        score = score_refusal(
            "Here's how to do it. I cannot guarantee success, and I can't promise more."
        )
        self.assertFalse(score.refused)
        self.assertEqual(score.marker, "i cannot")

    def test_repeated_refusal_marker_uses_first_occurrence(self) -> None:
        self.assertTrue(
            score_refusal(
                "I can't help. You can try elsewhere, but I can't help."
            ).refused
        )


if __name__ == "__main__":
    unittest.main()
