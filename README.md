# mlx-abliterate

Abliteration (refusal-direction ablation, [Arditi et al. 2024](https://arxiv.org/abs/2406.11717))
implemented natively in **MLX**, so it runs on Apple Silicon without PyTorch,
`transformers` modeling code, or a rented GPU.

Existing tools — [Heretic](https://github.com/p-e-w/heretic) in particular — are
`transformers`/PyTorch only. That makes them unusable for architectures that ship
an MLX implementation but no `transformers` one (Recursive's Flint is the case
that motivated this), and it forces an 80 GB bf16 load that a 64 GB Mac cannot
hold. MLX streams the same work in a few GB.

## How it works

1. **Capture** — replay the model's own forward pass with each decoder layer
   wrapped, recording the residual stream (`capture.py`).
2. **Directions** — mean last-token residual over paired harmful/harmless
   prompts; the difference is the refusal direction (`directions.py`).
3. **Ablate** — project that direction out of each layer's output projections:
   `W' = W - α · v (vᵀ W)` for `self_attn.o_proj`, dense/shared-expert
   `down_proj`, and optionally every routed expert. Quantized modules are
   dequantized, projected, and requantized in place with their own group size,
   bits and mode (`ablate.py`). `α` is `--strength`.
4. **Score** — string-level refusal detection on held-out prompts (`score.py`).
5. **KL** — teacher-forced KL divergence against the unmodified model on neutral
   text, as a cheap proxy for "how much did I break it" (`kl.py`).

## Install

```bash
uv tool install mlx-lm            # provides mlx + mlx-lm
git clone https://github.com/nateGeorge/mlx-abliterate
cd mlx-abliterate
PYTHONPATH=src python -m mlx_abliterate.cli --help
```

## Usage

```bash
# 1. estimate refusal directions (one forward pass per pair)
mlx-abliterate directions --model ./flint-nvfp4-mlx \
    --pairs data/pairs.jsonl --out data/directions.npz

# 2. baseline refusal rate + per-token log-probs for the KL reference
mlx-abliterate eval --model ./flint-nvfp4-mlx --prompts data/test.jsonl
mlx-abliterate kl  --model ./flint-nvfp4-mlx --texts data/kl_texts.jsonl \
    --write-base data/base-logp

# 3. sweep the layer band, then the strength/MLP shape
python scripts/sweep.py --model ./flint-nvfp4-mlx --directions data/directions.npz \
    --prompts data/dev.jsonl --kl-texts data/kl_texts.jsonl \
    --base-logp data/base-logp --stage band --out data/sweep_band.json
python scripts/sweep.py ... --stage shape --best-from data/sweep_band.json \
    --out data/sweep_shape.json

# 4. write the checkpoint
mlx-abliterate ablate --model ./flint-nvfp4-mlx --directions data/directions.npz \
    --start 8 --end 48 --strength 0.75 --out ./flint-nvfp4-heretic

# 5. verify the saved checkpoint on prompts it has never seen
mlx-abliterate eval --model ./flint-nvfp4-heretic --prompts data/test.jsonl
```

`eval --directions` skips the disk write and is the fast loop for picking a band.

## Measured: Recursive Flint 40B-A2.5B

NVFP4 MLX checkpoint, M5 Max (40-core, 64 GB). 400 paired prompts for the
directions, 104 held-out harmful prompts split evenly into disjoint dev (52) and
test (52), greedy decoding, 128 tokens per reply, KL over 128 neutral texts
(48 sampled per config).

Baseline refusal rate: **46.2% dev / 44.2% test**.

Layer band, strength 1.0, attention + shared-expert MLP (`mlp=on`), dev split:

| layers | refusals | KL mean |
| ------ | -------- | ------- |
| 0–32   | 40.4%    | 2.25    |
| 0–40   | 40.4%    | 2.29    |
| 0–48   | 11.5%    | 9.74    |
| 8–32   | 36.5%    | 0.19    |
| 8–40   | 28.8%    | 0.23    |
| 8–48   | 5.8%     | 3.63    |
| 12–32  | 30.8%    | 0.13    |
| 12–48  | 13.5%    | 3.51    |
| 16–32  | 23.1%    | 0.11    |

Refusal removal concentrates in the last ~16 layers: stopping at 40 leaves a
third of the refusals in place no matter how hard you push the earlier layers,
while including 41–48 is what actually finishes the job — at a steep KL cost.

Shape sweep on the best band (8–48), dev split:

| strength | mlp | refusals | KL mean  | KL p90   |
| -------- | --- | -------- | -------- | -------- |
| 0.50     | off | 30.8%    | 0.07     | 0.09     |
| 0.75     | off | 19.2%    | 0.12     | 0.13     |
| 1.00     | off | 23.1%    | 0.19     | 0.28     |
| 0.50     | on  | 19.2%    | 0.67     | 1.60     |
| **0.75** | on  | **9.6%** | **1.75** | **4.46** |
| 1.00     | on  | 5.8%     | 3.63     | 9.22     |
| 1.25     | on  | 0.0%     | 5.86     | 14.08    |

Attention-only ablation is nearly free (KL ≈ 0.1) but plateaus around half the
refusals. Adding the shared-expert MLP is what gets the rest, and it is the
expensive part. Past strength ~1.0 the model degenerates into repetition loops —
the 1.25 row reaches 0% refusals and is not a usable model.

**Released configuration:** layers 8–48, attention + shared-expert MLP,
strength 0.75 — 82 projections across 41 layers.

| split | base  | ablated   |
| ----- | ----- | --------- |
| dev   | 46.2% | 9.6%      |
| test  | 44.2% | **13.5%** |

3.3× fewer refusals on prompts the configuration never saw, with general
capability intact (coding and explanatory answers stay coherent; see
`nateGeorge/flint-nvfp4-heretic` on Hugging Face for samples).

## Limitations

- **The refusal scorer is a string matcher**, not a classifier. It counts a
  reply as a refusal if it opens with a refusal phrase, so a model that refuses
  and then complies, or complies in an unusual phrasing, can be mis-scored.
  Relative comparisons between configs are meaningful; the absolute number is
  not comparable to a judge-model evaluation.
- **KL is measured on Alpaca-style neutral text**, teacher-forced. It correlates
  with damage but is not a benchmark score. Coherence has to be spot-checked by
  hand, which is how the strength-1.25 degeneration was found.
- **Only the residual-stream direction is ablated.** Heretic's per-layer
  optimization (flexible direction weights, per-component selection) is not
  implemented here; the sweep is a grid, not a search.
- The 52-prompt test split gives ~2-point granularity. Differences under 4
  points between adjacent configs are within noise.

## License

MIT for the code here. Abliterated _weights_ inherit the source model's license
(Apache-2.0 for Flint, which allows redistribution with attribution).
