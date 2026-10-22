# Handoff: finish `mlx-abliterate` and produce an abliterated Flint

You are taking over a working but unpublished implementation. Your job is to
turn a promising prototype into a measured release: a swept ablation with a KL
budget, a real held-out evaluation, and a saved MLX checkpoint that runs in
`mlx_lm` and can be re-packed for `rec`.

## Goal

Produce `flint-nvfp4-heretic` (or similarly named) — Recursive Flint 40B-A2.5B
with the refusal direction ablated — with published refusal-rate and
quality-regression numbers, from the checkpoint at
`/Users/nate/models/flint-nvfp4-mlx`.

## Current state (already verified, do not rebuild)

Repository: `github.com/nateGeorge/mlx-abliterate` (private), local checkout
`~/github/mlx-abliterate`. Python 3.12 environment that has MLX:
`~/.local/share/uv/tools/mlx-lm/bin/python3` (run with `PYTHONPATH=src`).

Modules:

- `src/mlx_abliterate/capture.py` — wraps decoder-layer classes to record the
  residual stream during a normal forward pass. Works on Flint (49 layers).
- `src/mlx_abliterate/directions.py` — mean last-token residual difference over
  paired harmful/harmless prompts; `rank_layers` scores layers by cosine.
- `src/mlx_abliterate/ablate.py` — `W' = W - v (v^T W)` applied to
  `self_attn.o_proj`, dense/shared-expert `down_proj`, optionally every routed
  expert. Dequantizes `QuantizedLinear`/`QuantizedSwitchLinear`, projects, and
  requantizes with the module's own group size, bits, mode and global scale.
- `src/mlx_abliterate/score.py` — string-level refusal detection.
- `src/mlx_abliterate/cli.py` — `directions`, `ablate`, `eval` subcommands.

Measured on 8 held-out prompts, 80-token replies, attention-only ablation of
layers 10–46: **38% → 12% refusals**. That is the number to beat and to verify
with a bigger sample.

## Known gaps (this is your task list)

1. **`ablate --out` does not write a valid checkpoint.** It flattens parameter
   names incorrectly. Use `mlx_lm.utils.save` / the `save_weights` + `quantize`
   path in `mlx_lm`, or mirror `mlx_lm.convert`'s output layout: copy
   `config.json`, `tokenizer*`, `chat_template.jinja`, `flint.py` from the source
   directory, then `mx.save_safetensors("model.safetensors", flattened)`. Verify
   by loading the result back with `mlx_lm.load(..., trust_remote_code=True)`
   and generating.
2. **No parameter sweep.** Layer band, per-layer strength, and which projections
   to ablate are hard-coded. Sweep at least: start ∈ {0, 8, 12, 16},
   end ∈ {32, 40, 48}, attention-only vs attention+shared-expert, strength ∈
   {0.5, 0.75, 1.0, 1.25}. Objective: minimize refusals subject to a KL budget.
3. **No KL measurement.** Add KL(harmless completions ‖ base) over a validation
   slice; Heretic-style target is KL < 0.3 for a good trade.
4. **Refusal scoring is string matching.** Keep it as the primary metric for
   comparability, but add an optional judge (any OpenAI-compatible endpoint) and
   report both.
5. **Prompt set is 8 pairs.** Build a real set (see below), with a strict
   train/holdout split — the current holdout is only 8 prompts.
6. **`moe_experts` path untested.** It iterates routed-expert tensors; check it
   does not blow memory and that requantization error stays acceptable.

## Public data to use

- `mlabonne/harmful_behaviors` — 500 harmful instructions (AdvBench-derived),
  the standard side of the pair for abliteration.
- `mlabonne/harmless_alpaca` — 500 benign Alpaca instructions, used as the
  harmless side.
- `tatsu-lab/alpaca` — full Alpaca, for building additional harmless prompts and
  for KL validation completions.

Build ~300–500 pairs for direction estimation and keep 100+ prompts untouched as
holdout. Do not tune anything against the holdout.

## Runtime on this machine (M5 Max, 40-core, 64 GB)

- Direction estimation: one forward pass per prompt. ~0.2 s per short prompt, so
  1,000 prompts ≈ 3–5 minutes.
- Attention-only ablation: 49 layers, sub-second.
- Routed-expert ablation (`--moe-experts`): 12,032 tensors to dequantize and
  requantize — expect 15–40 minutes.
- Evaluation: 100 prompts × 200 tokens at ~85 tok/s ≈ 4 minutes per configuration.
- A 25–40 configuration sweep with KL: **2–4 hours total.** Overnight is more
  than enough; this does not need days and does not need a cloud GPU.

Memory: the 25 GB NVFP4 checkpoint fits with room to spare; ablation touches one
layer's tensors at a time, so peak stays under ~30 GB.

## Constraint

Every claim in the final README must come from a command you actually ran, with
the output shown. Do not report Heretic's published numbers as ours.
