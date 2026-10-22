"""Build the prompt sets used for direction estimation, evaluation and KL.

Everything comes from the public datasets the handoff names, pulled through the
Hugging Face datasets-server so no extra dependency is needed:

* ``mlabonne/harmful_behaviors`` - AdvBench-derived harmful instructions. The
  ``train`` split builds directions; the ``test`` split is the held-out
  evaluation set and is never used for tuning.
* ``mlabonne/harmless_alpaca`` - benign instructions, the harmless side of the
  pair, count-matched to the harmful side.
* ``tatsu-lab/alpaca`` - full instruction/response text for the KL validation
  slice, so KL is measured on realistic completions rather than bare prompts.

    python scripts/build_data.py --out data
"""

from __future__ import annotations

import argparse
import json
import random
import urllib.parse
import urllib.request
from pathlib import Path

ROWS_URL = "https://datasets-server.huggingface.co/rows"
PAGE = 100


def fetch_rows(dataset: str, split: str, want: int) -> list[dict]:
    """Page through a dataset split until ``want`` rows are collected."""
    rows: list[dict] = []
    offset = 0
    while len(rows) < want:
        query = urllib.parse.urlencode(
            {
                "dataset": dataset,
                "config": "default",
                "split": split,
                "offset": offset,
                "length": min(PAGE, want - len(rows)),
            }
        )
        try:
            with urllib.request.urlopen(f"{ROWS_URL}?{query}", timeout=60) as response:
                payload = json.load(response)
        except (OSError, json.JSONDecodeError) as exc:
            raise SystemExit(
                f"cannot fetch {dataset} ({split}) at offset {offset}: {exc}"
            ) from exc
        batch = payload.get("rows", [])
        if not batch:
            break
        rows.extend(row["row"] for row in batch)
        offset += len(batch)
    return rows[:want]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data")
    parser.add_argument("--pairs", type=int, default=400)
    parser.add_argument("--kl-texts", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    random.seed(args.seed)
    out = Path(args.out)

    harmful = [
        row["text"].strip()
        for row in fetch_rows("mlabonne/harmful_behaviors", "train", args.pairs)
    ]
    holdout = [
        row["text"].strip()
        for row in fetch_rows("mlabonne/harmful_behaviors", "test", 200)
    ]
    harmless_pool = [
        row["text"].strip()
        for row in fetch_rows("mlabonne/harmless_alpaca", "train", args.pairs * 2)
    ]
    harmless = random.sample(harmless_pool, len(harmful))

    pairs = [
        {"harmful": bad, "harmless": good}
        for bad, good in zip(harmful, harmless, strict=True)
    ]
    write_jsonl(out / "pairs.jsonl", pairs)
    write_jsonl(out / "holdout.jsonl", [{"prompt": prompt} for prompt in holdout])

    alpaca = [
        row["text"].strip()
        for row in fetch_rows("tatsu-lab/alpaca", "train", args.kl_texts * 4)
        if len(row["text"].strip()) > 200
    ]
    write_jsonl(
        out / "kl_texts.jsonl",
        [
            {"text": text}
            for text in random.sample(alpaca, min(args.kl_texts, len(alpaca)))
        ],
    )

    print(f"pairs       {len(pairs)}")
    print(f"holdout     {len(holdout)}")
    print(f"kl_texts    {min(args.kl_texts, len(alpaca))}")


if __name__ == "__main__":
    main()
