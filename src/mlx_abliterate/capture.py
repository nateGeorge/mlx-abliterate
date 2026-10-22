"""Layer-wise hidden-state capture for MLX language models.

The only universal interface we rely on is ``model.layers``, which every
``mlx_lm`` architecture exposes (the top-level ``Model`` holds the transformer
as ``model.model``, and its ``.layers`` property forwards). Each layer is
called as ``layer(h, mask, cache)`` in most architectures, but signatures vary,
so we prefer to replay the model's own forward pass and record the residual
stream by wrapping each layer in a recorder. That avoids reimplementing masking
(especially sliding-window attention) per architecture.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import mlx.core as mx


def transformer_of(model):
    """Return the module that owns ``.layers`` and ``.embed_tokens``."""
    for attr in ("model", "transformer", "language_model"):
        sub = getattr(model, attr, None)
        if sub is not None and hasattr(sub, "layers"):
            return sub
    if hasattr(model, "layers"):
        return model
    raise AttributeError("could not locate the transformer submodule holding .layers")


@dataclass
class Recorder:
    """Collect residual-stream outputs, one entry per decoder layer.

    MLX ``nn.Module`` overrides ``__setattr__`` (it registers parameters and
    children), so assigning a bound function onto a layer instance is silently
    ignored. We patch the layer *class* instead and restore it on exit.
    """

    hidden: list[mx.array] = field(default_factory=list)

    def install(self, transformer):
        self.transformer = transformer
        layer_types = {type(layer) for layer in transformer.layers}
        self._originals = {t: t.__call__ for t in layer_types}
        rec = self

        for layer_type, original in self._originals.items():

            def wrapper(self_layer, *args, _original=original, **kwargs):
                out = _original(self_layer, *args, **kwargs)
                rec.hidden.append(out)
                return out

            layer_type.__call__ = wrapper
        return self

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        for layer_type, original in self._originals.items():
            layer_type.__call__ = original
        return False

    def reset(self):
        self.hidden = []


def token_ids(tokenizer, text: str, add_generation_prompt: bool = False) -> mx.array:
    """Tokenize ``text`` as a single user turn for chat models."""
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            text = tokenizer.apply_chat_template(
                [{"role": "user", "content": text}],
                add_generation_prompt=add_generation_prompt,
                tokenize=False,
            )
        except (AttributeError, ValueError, TypeError):
            pass
    ids = tokenizer.encode(text)
    return mx.array([ids])


def last_token_states(model, recorder: Recorder, ids: mx.array) -> list[mx.array]:
    """Run a forward pass and return each layer's last-token residual (fp32)."""
    recorder.reset()
    out = model(ids)
    mx.eval(out, *recorder.hidden)
    return [h[0, -1].astype(mx.float32) for h in recorder.hidden]
