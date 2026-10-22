"""Permanent weight ablation for MLX models, including quantized weights.

Ablating an out-projection with unit direction ``v`` in its output space means

    y' = y - (y . v) v  =  ((I - v v^T) W) h

so the weight edit is ``W' = W - outer(v, W^T v)``. Quantized modules cannot be
edited in place: the packed ``uint32`` weights encode four values per word, so
we dequantize the module, apply the projection on the dense tensor, and
requantize with the module's own parameters (group size, bits, mode, scales).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import mlx.core as mx
from mlx.nn import QuantizedLinear

try:  # MoE expert stacks live in mlx_lm, not mlx.nn
    from mlx_lm.models.switch_layers import QuantizedSwitchLinear
except ImportError:  # pragma: no cover - mlx_lm not installed
    QuantizedSwitchLinear = None


QUANTIZED = tuple(t for t in (QuantizedLinear, QuantizedSwitchLinear) if t is not None)


@dataclass
class AblationReport:
    applied: int = 0
    skipped: int = 0
    layers: list[int] = field(default_factory=list)

    def summary(self) -> str:
        return f"ablated {self.applied} projection(s) across layers {self.layers} ({self.skipped} left unchanged)"


def _project(weight: mx.array, direction: mx.array) -> mx.array:
    """Remove ``direction`` from the row space of ``weight`` (shape [out, in])."""
    v = direction.reshape(-1).astype(weight.dtype)
    coeff = (v @ weight).astype(weight.dtype)  # [in] = v^T W
    return weight - v[:, None] * coeff[None, :]


def _project_quantized(module, direction: mx.array) -> None:
    """Dequantize, project, requantize a quantized linear module in place."""
    weight = module.weight
    scales = module.scales
    mode = getattr(module, "mode", "affine")
    bits = getattr(module, "bits", 4)
    group_size = getattr(module, "group_size", 64)
    global_scale = getattr(module, "global_scale", None)

    dense = mx.dequantize(
        weight,
        scales,
        biases=getattr(module, "biases", None),
        group_size=group_size,
        bits=bits,
        mode=mode,
        global_scale=global_scale,
    )
    projected = _project(dense, direction)
    new_weight, new_scales, new_biases = mx.quantize(
        projected,
        group_size=group_size,
        bits=bits,
        mode=mode,
        global_scale=global_scale,
    )
    module.weight = new_weight
    module.scales = new_scales
    if getattr(module, "biases", None) is not None:
        module.biases = new_biases


def _ablate_module(module, direction: mx.array, report: AblationReport) -> None:
    if isinstance(module, QUANTIZED):
        _project_quantized(module, direction)
    elif hasattr(module, "weight") and module.weight.dtype != mx.uint32:
        module.weight = _project(module.weight, direction)
    else:
        report.skipped += 1
        return
    report.applied += 1


def _down_projection_of(mlp):
    """Yield the output projections of an MLP block (dense, MoE, or shared)."""
    for attr in ("down_proj", "shared_expert", "experts"):
        sub = getattr(mlp, attr, None)
        if sub is None:
            continue
        if hasattr(sub, "down_proj"):
            yield sub.down_proj
        elif hasattr(sub, "down_proj") and hasattr(sub, "num_experts"):
            yield sub


def ablate_model(
    model,
    directions,
    start_layer: int = 0,
    end_layer: int | None = None,
    attention: bool = True,
    mlp: bool = False,
    moe_experts: bool = False,
) -> AblationReport:
    """Project the refusal direction out of the model's output projections.

    ``attention`` ablates ``self_attn.o_proj``; ``mlp`` ablates dense and shared
    expert ``down_proj``; ``moe_experts`` additionally touches every routed
    expert (expensive: one dequant/requant per expert tensor).
    """
    report = AblationReport()
    layers = model.layers
    end_layer = len(layers) - 1 if end_layer is None else end_layer

    for index in range(start_layer, min(end_layer, len(layers) - 1) + 1):
        direction = directions.vectors[index]
        layer = layers[index]
        touched = False

        attn = getattr(layer, "self_attn", None)
        if attention and attn is not None and hasattr(attn, "o_proj"):
            _ablate_module(attn.o_proj, direction, report)
            touched = True

        block = getattr(layer, "mlp", None)
        if mlp and block is not None:
            if hasattr(block, "down_proj"):
                _ablate_module(block.down_proj, direction, report)
                touched = True
            shared = getattr(block, "shared_expert", None)
            if shared is not None and hasattr(shared, "down_proj"):
                _ablate_module(shared.down_proj, direction, report)
                touched = True
            experts = getattr(block, "switch_mlp", None) or getattr(
                block, "experts", None
            )
            if moe_experts and experts is not None:
                for name in ("down_proj", "gate_proj", "up_proj"):
                    module = getattr(experts, name, None)
                    if module is not None:
                        _ablate_module(module, direction, report)

        if touched:
            report.layers.append(index)

    mx.eval(model.parameters())
    return report
