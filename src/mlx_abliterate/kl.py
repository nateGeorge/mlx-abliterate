"""KL divergence between the base model and an ablated model.

The refusal rate says whether the model still refuses; it says nothing about
how much of the original distribution survived. We measure the standard
Heretic-style trade-off:

    KL( base || ablated ) = E_{x ~ base}[ log p_base(x) - log p_ablated(x) ]

evaluated teacher-forced over fixed token sequences, so it is deterministic and
needs no sampling. Two passes are required because a 25 GB checkpoint cannot be
held twice on a 64 GB machine: the first pass stores the base model's
log-probabilities on disk, the second loads them and scores the ablated model
against them.
"""

from __future__ import annotations

import json
from pathlib import Path

import mlx.core as mx
import numpy as np


def _tokenize(tokenizer, text: str, max_tokens: int) -> list[int]:
    ids = tokenizer.encode(text)
    return ids[:max_tokens]


def _log_softmax_rows(logits: mx.array) -> mx.array:
    """log softmax over the vocabulary axis for a [tokens, vocab] tensor."""
    shifted = logits - mx.max(logits, axis=-1, keepdims=True)
    return shifted - mx.log(mx.sum(mx.exp(shifted), axis=-1, keepdims=True))


def _forward_logprobs(model, ids: list[int]) -> np.ndarray:
    """Teacher-forced log p(token_t | token_<t) for t = 1..T-1."""
    inputs = mx.array([ids])
    logits = model(inputs)[0, :-1].astype(mx.float32)
    logp = _log_softmax_rows(logits)
    mx.eval(logp)
    return np.array(logp.astype(mx.float16))


def write_base_logprobs(model, tokenizer, texts, out_dir: str, max_tokens: int) -> int:
    """Store base log-probabilities (and the token ids used) for a later KL pass."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    meta = []
    written = 0
    for index, text in enumerate(texts):
        ids = _tokenize(tokenizer, text, max_tokens)
        if len(ids) < 4:
            meta.append(None)
            continue
        np.save(out / f"{index:04d}.npy", _forward_logprobs(model, ids))
        meta.append(ids)
        written += 1
    (out / "meta.json").write_text(json.dumps(meta))
    return written


def compare_kl(model, tokenizer, texts, base_dir: str, max_tokens: int) -> dict:
    """Mean/p50/p90 per-token KL(base || ablated) against stored base log-probs."""
    base = Path(base_dir)
    try:
        meta = json.loads((base / "meta.json").read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"cannot read base log-probs from {base}: {exc}") from exc
    per_token: list[np.ndarray] = []
    for index, _text in enumerate(texts):
        stored_ids = meta[index] if index < len(meta) else None
        if not stored_ids:
            continue
        base_logp = np.load(base / f"{index:04d}.npy").astype(np.float32)
        current = _forward_logprobs(model, stored_ids).astype(np.float32)
        length = min(len(base_logp), len(current))
        base_logp, current = base_logp[:length], current[:length]
        # KL(base || ablated), summed over the vocabulary per position.
        per_token.append(np.sum(np.exp(base_logp) * (base_logp - current), axis=-1))
    if not per_token:
        return {"tokens": 0, "kl_mean": None}
    flat = np.concatenate(per_token)
    return {
        "tokens": int(flat.size),
        "kl_mean": float(flat.mean()),
        "kl_p50": float(np.percentile(flat, 50)),
        "kl_p90": float(np.percentile(flat, 90)),
        "kl_max": float(flat.max()),
    }
