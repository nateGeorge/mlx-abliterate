"""Permanent weight ablation for MLX models, including quantized weights.

Ablating an out-projection with unit direction ``v`` in its output space means

    y' = y - s (y . v) v  =  ((I - s v v^T) W) h

so the weight edit is ``W' = W - s outer(v, W^T v)``. ``s`` is the ablation
strength (1.0 removes the direction completely, 0.5 halves it).

Quantized modules cannot be edited in place: the packed ``uint32`` weights
encode several values per word, so we dequantize the module, apply the
projection on the dense tensor, and requantize with the module's own
parameters (group size, bits, mode, per-tensor or per-expert global scale).

Only projections whose *output* width equals the residual width can be
ablated, because the refusal direction lives in the residual stream. For
Flint that is ``self_attn.o_proj``, dense/shared ``down_proj`` and routed-expert
``down_proj``; ``gate_proj``/``up_proj`` write to the intermediate width and are
skipped automatically.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import mlx.core as mx
from mlx.nn import QuantizedLinear

try:  # MoE expert stacks live in mlx_lm, not mlx.nn
    from mlx_lm.models.switch_layers import QuantizedSwitchLinear  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - mlx_lm not installed
    QuantizedSwitchLinear = None


QUANTIZED = tuple(t for t in (QuantizedLinear, QuantizedSwitchLinear) if t is not None)


@dataclass
class AblationReport:
    applied: int = 0
    skipped: int = 0
    mismatched: int = 0
    layers: list[int] = field(default_factory=list)

    def summary(self) -> str:
        return (
            f"ablated {self.applied} projection(s) across {len(self.layers)} layer(s) "
            f"[{self.layers[0]}-{self.layers[-1]}]"
            if self.layers
            else "ablated 0 projection(s)"
        ) + (
            f"; skipped {self.skipped} unquantized-or-unknown, "
            f"{self.mismatched} with a width mismatch"
        )


def _out_width(module) -> int:
    """Output width of a linear-ish module, packed or dense."""
    shape = module.weight.shape
    # QuantizedSwitchLinear stacks experts: [num_experts, out, in_packed]
    return shape[1] if len(shape) == 3 else shape[0]


def _project(weight: mx.array, direction: mx.array, strength: float) -> mx.array:
    """Remove ``strength`` x ``direction`` from the row space of ``weight``."""
    v = direction.reshape(-1).astype(weight.dtype)
    if weight.ndim == 2:  # [out, in]
        coeff = (v @ weight).astype(weight.dtype)  # [in]
        return weight - (strength * v)[:, None] * coeff[None, :]
    # [experts, out, in] - one shared direction across the expert stack
    coeff = mx.einsum("o,eoi->ei", v, weight).astype(weight.dtype)  # [experts, in]
    return weight - (strength * v)[None, :, None] * coeff[:, None, :]


def _project_quantized(module, direction: mx.array, strength: float) -> None:
    """Dequantize, project, requantize a quantized linear module in place."""
    bits = getattr(module, "bits", 4)
    group_size = getattr(module, "group_size", 64)
    mode = getattr(module, "mode", "affine")
    global_scale = getattr(module, "global_scale", None)
    biases = getattr(module, "biases", None)

    dense = mx.dequantize(
        module.weight,
        module.scales,
        biases=biases,
        group_size=group_size,
        bits=bits,
        mode=mode,
        global_scale=global_scale,
    )
    projected = _project(dense, direction, strength)
    del dense

    kwargs = {"group_size": group_size, "bits": bits, "mode": mode}
    if global_scale is not None:
        kwargs["global_scale"] = global_scale
    packed = mx.quantize(projected, **kwargs)
    del projected

    module.weight = packed[0]
    module.scales = packed[1]
    if biases is not None and len(packed) > 2:
        module.biases = packed[2]
    mx.eval(module.weight, module.scales)


def _ablate_module(
    module, direction: mx.array, strength: float, report: AblationReport
) -> None:
    if module is None:
        return
    if _out_width(module) != direction.shape[-1]:
        report.mismatched += 1
        return
    if isinstance(module, QUANTIZED):
        _project_quantized(module, direction, strength)
    elif hasattr(module, "weight") and module.weight.dtype != mx.uint32:
        module.weight = _project(module.weight, direction, strength)
        mx.eval(module.weight)
    else:
        report.skipped += 1
        return
    report.applied += 1


def ablate_model(
    model,
    directions,
    start_layer: int = 0,
    end_layer: int | None = None,
    attention: bool = True,
    mlp: bool = False,
    moe_experts: bool = False,
    strength: float = 1.0,
) -> AblationReport:
    """Project the refusal direction out of the model's residual-writing projections.

    ``attention`` ablates ``self_attn.o_proj``; ``mlp`` ablates dense and shared
    expert ``down_proj``; ``moe_experts`` additionally touches every routed
    expert ``down_proj`` (expensive: one dequant/requant per expert tensor).
    """
    report = AblationReport()
    layers = model.layers
    end_layer = len(layers) - 1 if end_layer is None else end_layer

    for index in range(start_layer, min(end_layer, len(layers) - 1) + 1):
        direction = directions.vectors[index]
        layer = layers[index]
        touched = False

        attn = getattr(layer, "self_attn", None)
        if attention and attn is not None:
            _ablate_module(attn.o_proj, direction, strength, report)
            touched = True

        block = getattr(layer, "mlp", None)
        if mlp and block is not None:
            if getattr(block, "down_proj", None) is not None:
                _ablate_module(block.down_proj, direction, strength, report)
                touched = True
            shared = getattr(block, "shared_expert", None)
            if shared is not None:
                _ablate_module(shared.down_proj, direction, strength, report)
                touched = True
            experts = getattr(block, "switch_mlp", None)
            if moe_experts and experts is not None:
                _ablate_module(experts.down_proj, direction, strength, report)
                touched = True

        if touched:
            report.layers.append(index)

    mx.eval(model.parameters())
    return report
