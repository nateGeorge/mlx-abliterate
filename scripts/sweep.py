"""Sweep ablation configurations and record refusal rate plus KL.

Every configuration runs in its own process so a 25 GB checkpoint is never held
twice and a crash cannot poison the rest of the sweep. Results are written to
``sweep.json`` as they finish, so the file is always usable.

    python scripts/sweep.py --model MODEL --directions data/directions.npz \
        --prompts data/holdout.jsonl --kl-texts data/kl_texts.jsonl \
        --base-logp data/base-logp --out data/sweep.json --stage band

Two stages exist because the full grid is too large to run whole:

* ``band``  - strength 1.0, attention + shared-expert MLP, over start/end bands.
* ``shape`` - the best band from stage one, over projection choice and strength.
"""

from __future__ import annotations

import argparse
import itertools
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE.parent / "src"


def load_json(path: str, what: str):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"cannot read {what} from {path}: {exc}") from exc


def band_configs() -> list[dict]:
    return [
        {
            "label": f"band s{start} e{end} mlp",
            "start": start,
            "end": end,
            "strength": 1.0,
            "mlp": True,
        }
        for start, end in itertools.product([0, 8, 12, 16], [32, 40, 48])
    ]


def shape_configs(best: dict) -> list[dict]:
    return [
        {
            "label": f"shape s{best['start']} e{best['end']} mlp={mlp} a={strength}",
            "start": best["start"],
            "end": best["end"],
            "strength": strength,
            "mlp": mlp,
        }
        for mlp, strength in itertools.product([False, True], [0.5, 0.75, 1.0, 1.25])
    ]


def score_config(args, config: dict) -> dict:
    """Load once, ablate, then measure refusal rate and KL for one configuration."""
    sys.path.insert(0, str(SRC))
    from mlx_abliterate.ablate import ablate_model
    from mlx_abliterate.cli import (
        generate_replies,
        load_directions,
        load_model,
        read_jsonl,
    )
    from mlx_abliterate.kl import compare_kl
    from mlx_abliterate.score import refusal_rate

    model, tokenizer = load_model(args.model)
    report = ablate_model(
        model,
        load_directions(args.directions),
        start_layer=config["start"],
        end_layer=config["end"],
        attention=True,
        mlp=config["mlp"],
        moe_experts=config.get("moe_experts", False),
        strength=config["strength"],
    )
    prompts = [row["prompt"] for row in read_jsonl(args.prompts)][: args.limit]
    replies = generate_replies(model, tokenizer, prompts, args.max_tokens)
    texts = [row["text"] for row in read_jsonl(args.kl_texts)][: args.kl_limit]
    kl = compare_kl(model, tokenizer, texts, args.base_logp, args.max_tokens)
    return {
        **config,
        "applied": report.applied,
        "mismatched": report.mismatched,
        "refusal_rate": refusal_rate(replies),
        "prompts": len(prompts),
        **kl,
    }


def run_one(args, config: dict) -> dict:
    """Score one configuration in a child process and return its record."""
    cmd = [
        sys.executable,
        str(HERE / "sweep.py"),
        "--model",
        args.model,
        "--directions",
        args.directions,
        "--prompts",
        args.prompts,
        "--kl-texts",
        args.kl_texts,
        "--base-logp",
        args.base_logp,
        "--limit",
        str(args.limit),
        "--max-tokens",
        str(args.max_tokens),
        "--kl-limit",
        str(args.kl_limit),
        "--single",
        json.dumps(config),
    ]
    try:
        completed = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)
    except subprocess.TimeoutExpired:
        return {**config, "error": "timeout after 7200s"}
    if completed.returncode != 0:
        tail = completed.stderr.strip().splitlines()[-1:] or ["no stderr"]
        return {**config, "error": tail[0]}
    try:
        return json.loads(completed.stdout.strip().splitlines()[-1])
    except (IndexError, json.JSONDecodeError) as exc:
        return {**config, "error": f"unparseable child output: {exc}"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--directions", required=True)
    parser.add_argument("--prompts", required=True)
    parser.add_argument("--kl-texts", required=True)
    parser.add_argument("--base-logp", required=True)
    parser.add_argument("--out")
    parser.add_argument("--stage", choices=["band", "shape"], default="band")
    parser.add_argument("--limit", type=int, default=48)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--kl-limit", type=int, default=48)
    parser.add_argument("--best-from", help="sweep.json from the band stage")
    parser.add_argument("--single", help="internal: JSON config to score here")
    args = parser.parse_args()

    if args.single:
        print(json.dumps(score_config(args, load_json_str(args.single))))
        return

    if args.stage == "band":
        configs = band_configs()
    else:
        if not args.best_from:
            raise SystemExit("--stage shape needs --best-from sweep.json")
        configs = shape_configs(load_json(args.best_from, "band results")["best"])

    if not args.out:
        raise SystemExit("--out is required unless --single is used")
    out = Path(args.out)
    results = []
    for config in configs:
        print(f"== {config['label']}", flush=True)
        record = run_one(args, config)
        results.append(record)
        print(f"   {json.dumps(record)}", flush=True)
        payload = {"stage": args.stage, "results": results}
        ok = [r for r in results if "refusal_rate" in r]
        if ok:
            payload["best"] = min(
                ok, key=lambda r: (r["refusal_rate"], r.get("kl_mean") or 0.0)
            )
        out.write_text(json.dumps(payload, indent=2))
    print(f"wrote {out}")


def load_json_str(value: str) -> dict:
    try:
        return json.loads(value)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"malformed --single config: {exc}") from exc


if __name__ == "__main__":
    main()
