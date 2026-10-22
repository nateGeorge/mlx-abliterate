# Handoff

**Status: complete.** The release shipped on 2026-10-22.

- Code: https://github.com/nateGeorge/mlx-abliterate (public, MIT)
- Weights: https://huggingface.co/nateGeorge/flint-nvfp4-heretic (Apache-2.0)

## What was delivered

Recursive Flint 40B-A2.5B (NVFP4 MLX) with the refusal direction ablated across
layers 8–48 — `self_attn.o_proj` plus shared-expert `down_proj`, 82 projections,
strength 0.75.

|                             | base  | released    |
| --------------------------- | ----- | ----------- |
| refusals, dev (52)          | 46.2% | 9.6%        |
| refusals, test (52, unseen) | 44.2% | 13.5%       |
| KL mean / p90               | —     | 1.75 / 4.46 |

The old prototype number (38% → 12% on 8 prompts, attention-only, layers 10–46)
is superseded: 8 prompts was noise, and attention-only ablation plateaus around
half the refusals. Full sweep tables are in the README.

## Verified, do not rebuild

- `ablate --out` writes a checkpoint that loads in a fresh process and generates.
- The Hub copy is byte-identical (LFS oid matches local sha256) and was loaded
  and run from the Hub id in a clean process.
- `tests/test_ablate.py` (8 tests) covers the projection math, the width guard,
  the 3-D expert stack, and the scorer.

## Where the remaining juice is

1. **Per-layer optimization instead of a grid.** Heretic searches flexible
   per-layer direction weights; this sweeps a band and a scalar. The frontier
   shows the whole tradeoff lives in layers 41–48, so weighting those
   individually is the obvious next win.
2. **A better refusal scorer.** A string matcher on the reply opening cannot see
   a model that refuses then complies. Every number here is relative, not
   absolute.
3. **A second direction.** Only one residual direction is removed. Arditi et al.
   and later work find multiple; one direction may be why strength 0.75 still
   leaves 13%.

Procedure and traps are written up in the `mlx-abliteration` skill
(`~/.pi/agent/skills/mlx-abliteration/SKILL.md`).
