"""Unit tests for the ablation math and the refusal scorer.

Run with:  PYTHONPATH=src python -m unittest discover -s tests -v
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import mlx.core as mx

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mlx_abliterate.ablate import _out_width, _project  # noqa: E402
from mlx_abliterate.score import score_refusal  # noqa: E402


def unit_vector(size: int, seed: int = 0) -> mx.array:
    vec = mx.random.normal((size,), key=mx.random.key(seed)).astype(mx.float32)
    return vec / mx.linalg.norm(vec)


class ProjectTest(unittest.TestCase):
    def test_full_strength_removes_the_direction(self) -> None:
        weight = mx.random.normal((32, 64), key=mx.random.key(1)).astype(mx.float32)
        direction = unit_vector(32)
        projected = _project(weight, direction, 1.0)
        residual = direction @ projected  # v^T W' should vanish
        self.assertLess(float(mx.max(mx.abs(residual))), 1e-4)

    def test_half_strength_leaves_half_the_projection(self) -> None:
        weight = mx.random.normal((32, 64), key=mx.random.key(2)).astype(mx.float32)
        direction = unit_vector(32)
        before = direction @ weight
        after = direction @ _project(weight, direction, 0.5)
        self.assertLess(float(mx.max(mx.abs(after - 0.5 * before))), 1e-4)

    def test_zero_strength_is_identity(self) -> None:
        weight = mx.random.normal((32, 64), key=mx.random.key(3)).astype(mx.float32)
        direction = unit_vector(32)
        self.assertLess(
            float(mx.max(mx.abs(_project(weight, direction, 0.0) - weight))), 1e-5
        )

    def test_expert_stack_projects_every_expert(self) -> None:
        weight = mx.random.normal((4, 32, 64), key=mx.random.key(4)).astype(mx.float32)
        direction = unit_vector(32)
        projected = _project(weight, direction, 1.0)
        residual = mx.einsum("o,eoi->ei", direction, projected)
        self.assertLess(float(mx.max(mx.abs(residual))), 1e-4)


class WidthTest(unittest.TestCase):
    def test_dense_and_stacked_widths(self) -> None:
        class Dense:
            weight = mx.zeros((48, 32))

        class Stacked:
            weight = mx.zeros((8, 48, 32))

        self.assertEqual(_out_width(Dense()), 48)
        self.assertEqual(_out_width(Stacked()), 48)


class ScoreTest(unittest.TestCase):
    def test_plain_refusal(self) -> None:
        self.assertTrue(score_refusal("I can't help with that request.").refused)

    def test_compliance(self) -> None:
        self.assertFalse(score_refusal("Here's how you would do it: step 1").refused)

    def test_refusal_before_compliance_still_counts(self) -> None:
        text = "I cannot assist with that. You can read more elsewhere."
        self.assertTrue(score_refusal(text).refused)


if __name__ == "__main__":
    unittest.main()
