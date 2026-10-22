# mlx-abliterate

Abliteration (refusal-direction ablation, [Arditi et al. 2024](https://arxiv.org/abs/2406.11717))
implemented natively in **MLX**, so it works on Apple Silicon without PyTorch,
`transformers` modeling code, or a rented GPU.

Existing tools — [Heretic](https://github.com/p-e-w/heretic) in particular — are
`transformers`/PyTorch only. That makes them unusable for architectures that ship
an MLX implementation but no `transformers` one (Recursive's Flint is the case
that motivated this), and it forces an 80 GB bf16 load that a 64 GB Mac cannot
hold. MLX can stream the same work in a few GB.

## How it works

1. **Capture** — replay the model's own forward pass with each decoder layer
   wrapped, recording the residual stream (`capture.py`).
2. **Directions** — mean last-token residual over paired harmful/harmless
   prompts; the difference is the refusal direction (`directions.py`).
3. **Ablate** — project that direction out of each layer's output projections:
   `W' = W - v (v^T W)` for `self_attn.o_proj`, dense/shared-expert `down_proj`,
   and optionally every routed expert. Quantized modules are dequantized,
   projected, and requantized in place with their own group size, bits and mode
   (`ablate.py`).
4. **Score** — string-level refusal detection on a held-out prompt set
   (`score.py`).

## Usage

```bash
# 1. estimate directions (one forward pass per prompt)
mlx-abliterate directions --model ./flint-nvfp4-mlx --pairs data/pairs.jsonl --out dirs.npz

# 2. measure the baseline refusal rate
mlx-abliterate eval --model ./flint-nvfp4-mlx --prompts data/holdout.jsonl

# 3. ablate in memory and re-measure
mlx-abliterate eval --model ./flint-nvfp4-mlx --prompts data/holdout.jsonl \
  --directions dirs.npz --start 10 --end 46
```

`ablate` writes a checkpoint directory when `--out` is given; `eval` with
`--directions` is the fast in-memory loop for picking the layer band.

## Measured

Recursive **Flint** 40B-A2.5B, NVFP4 MLX checkpoint, on an M5 Max (40-core,
64 GB). 8 prompt pairs for directions, 8 unseen prompts for scoring, 80 tokens
per reply, attention-only ablation across layers 10–46:

|         | refusals      |
| ------- | ------------- |
| base    | 38% (3/8)     |
| ablated | **12% (1/8)** |

Small sample and a blunt scorer — treat it as "the machinery works", not as a
published benchmark. The layer band and the set of projections are the two
knobs that still need a sweep; `--moe-experts` widens the ablation to every
routed expert at a large cost in dequant/requant time.

## License

MIT for the code here. Abliterated _weights_ inherit the source model's license
(Apache-2.0 for Flint, which allows redistribution with attribution).
