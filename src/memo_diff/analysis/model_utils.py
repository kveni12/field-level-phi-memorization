"""Paper-only helpers, retained from the research implementation."""

from __future__ import annotations


def _final_norm_and_head(model):
    raw = model.module if hasattr(model, "module") else model
    if hasattr(raw, "model") and hasattr(raw.model, "norm") and hasattr(raw, "lm_head"):
        return raw.model.norm, raw.lm_head
    if hasattr(raw, "transformer") and hasattr(raw.transformer, "ln_f") and hasattr(raw, "embed_out"):
        return raw.transformer.ln_f, raw.embed_out
    raise AttributeError("Could not locate final RMSNorm/LayerNorm and lm_head")
