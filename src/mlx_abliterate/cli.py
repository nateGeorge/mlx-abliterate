"""Command line: estimate refusal directions, ablate, score, and measure KL.

    mlx-abliterate directions --model MODEL --pairs pairs.jsonl --out dir.npz
    mlx-abliterate ablate     --model MODEL --directions dir.npz --start 8 --end 40 --out out-dir
    mlx-abliterate eval       --model MODEL --prompts holdout.jsonl
    mlx-abliterate kl         --model MODEL --texts texts.jsonl --write-base base-logp/
    mlx-abliterate kl         --model MODEL --directions dir.npz --base base-logp/

Directions are stored as a single fp32 array of shape [layers, hidden].
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mlx.core as mx

from .ablate import ablate_model
from .capture import Recorder, last_token_states, token_ids, transformer_of
from .directions import Directions, rank_layers, refusal_directions
from .kl import compare_kl, write_base_logprobs
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


def load_directions(path: str) -> Directions:
    """Read an npz written by ``mx.savez`` regardless of mlx's return shape."""
    loaded = mx.load(path)
    if isinstance(loaded, tuple):  # older mlx: (arrays, metadata)
        loaded = loaded[0]
    stored = loaded["directions"] if isinstance(loaded, dict) else loaded
    return Directions(vectors=[stored[i] for i in range(stored.shape[0])])


def maybe_ablate(model, args) -> None:
    """Apply the ablation described by the CLI flags, if any were given."""
    if not getattr(args, "directions", None):
        return
    report = ablate_model(
        model,
        load_directions(args.directions),
        start_layer=args.start,
        end_layer=args.end,
        attention=not getattr(args, "no_attention", False),
        mlp=not getattr(args, "no_mlp", False),
        moe_experts=getattr(args, "moe_experts", False),
        strength=getattr(args, "strength", 1.0),
    )
    print(report.summary())


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
    from mlx_lm.utils import load_config, save  # type: ignore[import-not-found]

    model, tokenizer = load_model(args.model, not args.no_trust_remote_code)
    config = load_config(Path(args.model))
    maybe_ablate(model, args)
    if args.out:
        out = Path(args.out)
        if out.exists():
            raise SystemExit(f"{out} already exists; pick a new path")
        # mlx_lm's own writer: correct flat parameter names, sharding, index file,
        # tokenizer, config, and a copy of any *.py the model needs.
        save(out, args.model, model, tokenizer, config)
        print(f"wrote {out}")


def generate_replies(
    model, tokenizer, prompts: list[str], max_tokens: int
) -> list[str]:
    """Greedy batch generation - deterministic and much faster than one at a time."""
    from mlx_lm import batch_generate  # type: ignore[import-not-found]

    encoded = [
        tokenizer.encode(
            tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                add_generation_prompt=True,
                tokenize=False,
            )
        )
        for prompt in prompts
    ]
    response = batch_generate(
        model,
        tokenizer,
        encoded,
        max_tokens=max_tokens,
        completion_batch_size=args_completion_batch_size(),
        prefill_batch_size=4,
    )
    return [text.strip() for text in response.texts]


def args_completion_batch_size() -> int:
    import os

    return int(os.environ.get("MLX_ABLITERATE_BATCH", "16"))


def cmd_eval(args):
    model, tokenizer = load_model(args.model, not args.no_trust_remote_code)
    maybe_ablate(model, args)
    rows = read_jsonl(args.prompts)
    if args.limit:
        rows = rows[: args.limit]
    prompts = [row["prompt"] for row in rows]
    replies = generate_replies(model, tokenizer, prompts, args.max_tokens)
    rate = refusal_rate(replies)
    print(f"refusals: {rate * 100:.0f}% of {len(replies)}")
    for prompt, reply in zip(prompts, replies, strict=True):
        first = reply.splitlines()[0] if reply else ""
        print(f"  [{prompt[:40]}] -> {first[:90]}")
    if args.out:
        Path(args.out).write_text(
            json.dumps(
                {
                    "refusal_rate": rate,
                    "replies": list(zip(prompts, replies, strict=True)),
                },
                indent=2,
            )
        )
        print(f"wrote {args.out}")


def cmd_kl(args):
    model, tokenizer = load_model(args.model, not args.no_trust_remote_code)
    texts = [row["text"] for row in read_jsonl(args.texts)][: args.limit or None]
    maybe_ablate(model, args)
    if args.write_base:
        n = write_base_logprobs(
            model, tokenizer, texts, args.write_base, args.max_tokens
        )
        print(f"wrote {n} base log-probability files to {args.write_base}")
        return
    if not args.base:
        raise SystemExit("kl needs either --write-base or --base")
    result = compare_kl(model, tokenizer, texts, args.base, args.max_tokens)
    print(json.dumps(result, indent=2))


def add_ablation_flags(parser):
    parser.add_argument("--directions", help="ablate in memory before running")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int)
    parser.add_argument("--strength", type=float, default=1.0)
    parser.add_argument("--no-attention", action="store_true")
    parser.add_argument("--no-mlp", action="store_true")
    parser.add_argument("--moe-experts", action="store_true")


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
    a.add_argument("--out")
    add_ablation_flags(a)
    a.set_defaults(func=cmd_ablate)

    e = sub.add_parser("eval", help="measure refusal rate")
    e.add_argument("--model", required=True)
    e.add_argument("--prompts", required=True)
    e.add_argument("--max-tokens", type=int, default=160)
    e.add_argument("--limit", type=int)
    e.add_argument("--out")
    add_ablation_flags(e)
    e.set_defaults(func=cmd_eval)

    k = sub.add_parser("kl", help="KL(harmless completions || base)")
    k.add_argument("--model", required=True)
    k.add_argument("--texts", required=True)
    k.add_argument("--write-base", help="directory to write base log-probs into")
    k.add_argument("--base", help="directory of base log-probs to compare against")
    k.add_argument("--max-tokens", type=int, default=128)
    k.add_argument("--limit", type=int)
    add_ablation_flags(k)
    k.set_defaults(func=cmd_kl)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
