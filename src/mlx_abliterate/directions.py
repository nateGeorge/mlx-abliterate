"""Refusal-direction estimation (Abliteration, Arditi et al. 2024).

For each layer we take the mean residual-stream activation at the final token
over a set of *harmful* prompts and over a set of *harmless* prompts that share
the same surface form. The difference is the refusal direction: adding it pushes
the model toward refusal, subtracting it biases toward compliance.

Sign convention used throughout this package: we store ``v`` such that
``h <- h - (h . v) v`` removes the refusal behavior.
"""

from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx

from .capture import Recorder, last_token_states, token_ids


@dataclass
class Directions:
    vectors: list[mx.array]  # one unit vector per layer, fp32

    def array(self) -> mx.array:
        return mx.stack(self.vectors)


def mean_state(
    model, recorder: Recorder, tokenizer, prompts: list[str]
) -> list[mx.array]:
    """Average last-token residual per layer over ``prompts``."""
    acc: list[mx.array] | None = None
    for prompt in prompts:
        states = last_token_states(model, recorder, token_ids(tokenizer, prompt))
        if acc is None:
            acc = [mx.zeros_like(s) for s in states]
        acc = [a + s for a, s in zip(acc, states, strict=True)]
    assert acc is not None, "prompt set is empty"
    return [a / len(prompts) for a in acc]


def refusal_directions(
    model,
    recorder: Recorder,
    tokenizer,
    harmful: list[str],
    harmless: list[str],
) -> Directions:
    """Estimate one unit refusal direction per layer."""
    bad = mean_state(model, recorder, tokenizer, harmful)
    good = mean_state(model, recorder, tokenizer, harmless)
    vectors = []
    for b, g in zip(bad, good, strict=True):
        diff = b - g
        norm = mx.linalg.norm(diff)
        vectors.append(diff / mx.maximum(norm, 1e-6))
    mx.eval(*vectors)
    return Directions(vectors=vectors)


def rank_layers(
    directions: Directions, hidden_states: list[mx.array]
) -> list[tuple[int, float]]:
    """Score layers by |cos(h, v)| at the final token, descending.

    A layer whose refusal direction is strongly aligned with the residual
    stream is where ablation actually bites; uniform choices are the fallback
    when this signal is flat.
    """
    scores = []
    for index, (v, h) in enumerate(zip(directions.vectors, hidden_states, strict=True)):
        cosine = abs(float(mx.sum(h * v))) / max(float(mx.linalg.norm(h)), 1e-6)
        scores.append((index, cosine))
    return sorted(scores, key=lambda item: item[1], reverse=True)
