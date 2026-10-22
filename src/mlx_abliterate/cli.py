"""Command line: estimate refusal directions, ablate, and score the result.

    mlx-abliterate directions --model MODEL --pairs pairs.jsonl --out dir.npz
    mlx-abliterate ablate     --model MODEL --directions dir.npz --start 8 --end 40 --out out-dir
    mlx-abliterate eval       --model MODEL --prompts holdout.jsonl

Directions are stored as a single fp32 array of shape [layers, hidden].
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import mlx.core as mx

from .ablate import ablate_model
from .capture import Recorder, last_token_states, token_ids, transformer_of
from .directions import Directions, rank_layers, refusal_directions
from .score import refusal_rate


def read_jsonl(path: str) -> list[dict]:
    try:
        with open(path, encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]
    except OSError as exc:
        raise SystemExit(f"cannot read {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise SystemExit(f"malformed JSONL in {path}: {exc}") from exc


def load_model(model_path: str, trust_remote_code: bool = True):
    from mlx_lm import load  # type: ignore[import-not-found]

    return load(
        model_path,
        tokenizer_config={"trust_remote_code": trust_remote_code},
        trust_remote_code=trust_remote_code,
    )


def load_arrays(path: str):
    """Return the values of an npz written by ``mx.savez``, in insertion order."""
    from mlx_lm.models.cache import _BaseCache  # type: ignore[import-not-found]  # noqa: F401

    arrays = mx.load(path)
    return list(arrays.values())


def cmd_directions(args):
    model, tokenizer = load_model(args.model, not args.no_trust_remote_code)
    recorder = Recorder().install(transformer_of(model))
    pairs = read_jsonl(args.pairs)
    harmful = [p["harmful"] for p in pairs]
    harmless = [p["harmless"] for p in pairs]
    directions = refusal_directions(model, recorder, tokenizer, harmful, harmless)

    probe = last_token_states(model, recorder, token_ids(tokenizer, harmful[0]))
    for index, score in rank_layers(directions, probe)[: args.top]:
        print(f"layer {index:3d}  |h.v| = {score:.3f}")

    if args.out:
        mx.savez(args.out, directions=directions.array())
        print(f"wrote {args.out}")


def cmd_ablate(args):
    model, tokenizer = load_model(args.model, not args.no_trust_remote_code)
    stored = load_arrays(args.directions)[0]
    directions = Directions(vectors=[stored[i] for i in range(stored.shape[0])])
    report = ablate_model(
        model,
        directions,
        start_layer=args.start,
        end_layer=args.end,
        attention=not args.no_attention,
        mlp=not args.no_mlp,
        moe_experts=args.moe_experts,
    )
    print(report.summary())
    if args.out:
        Path(args.out).mkdir(parents=True, exist_ok=True)
        for item in sorted(os.listdir(args.model)):
            src = Path(args.model) / item
            dst = Path(args.out) / item
            if src.is_file() and not dst.exists():
                dst.write_bytes(src.read_bytes())
        weights = dict(model.parameters())
        flat = {}
        for key, value in weights.items():
            flat[key.replace(".", "/")] = value
        mx.save_safetensors(
            str(Path(args.out) / "model.safetensors"), flat, metadata={"format": "mlx"}
        )
        print(f"wrote {args.out}")


def cmd_eval(args):
    from mlx_lm import generate  # type: ignore[import-not-found]

    model, tokenizer = load_model(args.model, not args.no_trust_remote_code)
    if args.directions:
        stored = load_arrays(args.directions)[0]
        report = ablate_model(
            model,
            Directions(vectors=[stored[i] for i in range(stored.shape[0])]),
            start_layer=args.start,
            end_layer=args.end,
            moe_experts=args.moe_experts,
        )
        print(report.summary())
    prompts = [row["prompt"] for row in read_jsonl(args.prompts)]
    replies = []
    for prompt in prompts:
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            tokenize=False,
        )
        replies.append(
            generate(
                model, tokenizer, prompt=text, max_tokens=args.max_tokens, verbose=False
            )
        )
    rate = refusal_rate(replies)
    print(f"refusals: {rate * 100:.0f}% of {len(replies)}")
    for prompt, reply in zip(prompts, replies, strict=True):
        print(f"  [{prompt[:40]}] -> {reply.strip().splitlines()[0][:90]}")


def main(argv=None):
    parser = argparse.ArgumentParser(prog="mlx-abliterate")
    parser.add_argument("--no-trust-remote-code", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    d = sub.add_parser("directions", help="estimate refusal directions")
    d.add_argument("--model", required=True)
    d.add_argument("--pairs", required=True)
    d.add_argument("--out")
    d.add_argument("--top", type=int, default=10)
    d.set_defaults(func=cmd_directions)

    a = sub.add_parser("ablate", help="apply the ablation to model weights")
    a.add_argument("--model", required=True)
    a.add_argument("--directions", required=True)
    a.add_argument("--start", type=int, default=0)
    a.add_argument("--end", type=int)
    a.add_argument("--out")
    a.add_argument("--no-attention", action="store_true")
    a.add_argument("--no-mlp", action="store_true")
    a.add_argument("--moe-experts", action="store_true")
    a.set_defaults(func=cmd_ablate)

    e = sub.add_parser("eval", help="measure refusal rate")
    e.add_argument("--model", required=True)
    e.add_argument("--prompts", required=True)
    e.add_argument("--max-tokens", type=int, default=96)
    e.add_argument("--directions", help="ablate in memory before evaluating")
    e.add_argument("--start", type=int, default=0)
    e.add_argument("--end", type=int)
    e.add_argument("--moe-experts", action="store_true")
    e.set_defaults(func=cmd_eval)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
