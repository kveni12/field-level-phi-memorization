import torch
import torch.nn as nn
import pytest

from memo_diff.analysis.jacobian_lens import (
    apply_jacobian_lens,
    autotune_jac_batches,
    jacobian_for_ids,
    validate_lens_manifest,
    valid_position_mask,
)


def test_valid_position_mask_drops_sinks_and_last() -> None:
    mask = valid_position_mask(10, skip_first=3)
    assert mask.tolist() == [False, False, False, True, True, True, True, True, True, False]


def test_apply_rejects_missing_source_and_uses_identity_only_at_target() -> None:
    h0 = torch.randn(5, 4)
    h1 = torch.randn(5, 4)
    try:
        apply_jacobian_lens([h0, h1], jacobians={}, target_layer=1)
    except KeyError as exc:
        assert "missing Jacobian for source layer 0" in str(exc)
    else:
        raise AssertionError("missing source Jacobian should fail")

    out = apply_jacobian_lens(
        [h0, h1], jacobians={}, target_layer=1, layers=[1]
    )
    assert len(out) == 1
    assert torch.equal(out[0], h1)


def test_apply_sparse_fitted_layers_preserves_requested_order() -> None:
    hs = [torch.randn(3, 2) for _ in range(4)]
    jacobians = {0: torch.eye(2) * 2, 2: torch.eye(2) * 3}
    out = apply_jacobian_lens(
        hs, jacobians=jacobians, target_layer=3, layers=[0, 2, 3]
    )
    assert torch.allclose(out[0], hs[0] * 2)
    assert torch.allclose(out[1], hs[2] * 3)
    assert torch.equal(out[2], hs[3])


def test_cache_without_manifest_is_rejected() -> None:
    with pytest.raises(ValueError, match="no compatibility manifest"):
        validate_lens_manifest(
            None,
            model=_TinyLM(),
            tokenizer=object(),
            checkpoint_path="checkpoint",
            jacobians={},
            target_layer=2,
        )


class _Block(nn.Module):
    def __init__(self, scale: float) -> None:
        super().__init__()
        self.lin = nn.Linear(4, 4, bias=False)
        with torch.no_grad():
            self.lin.weight.copy_(torch.eye(4) * scale)

    def forward(self, hidden_states, **_kwargs):
        return self.lin(hidden_states)


class _TinyLM(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = type("C", (), {"hidden_size": 4, "num_hidden_layers": 2})()
        self.model = nn.Module()
        self.model.embed_tokens = nn.Embedding(20, 4)
        nn.init.normal_(self.model.embed_tokens.weight, std=0.3)
        self.model.layers = nn.ModuleList([_Block(2.0), _Block(3.0)])

    def forward(self, input_ids, use_cache=False, **_kwargs):
        h = self.model.embed_tokens(input_ids)
        for layer in self.model.layers:
            h = layer(h)
        return h


def test_jacobian_recovers_linear_map() -> None:
    torch.manual_seed(0)
    model = _TinyLM()
    ids = torch.randint(0, 20, (1, 12))
    js = jacobian_for_ids(model, ids, dim_batch=4, skip_first=2)
    got = js[1]
    want = torch.eye(4) * 3.0
    assert torch.allclose(got, want, atol=1e-4, rtol=1e-4)
    got0 = js[0]
    want0 = torch.eye(4) * 6.0
    assert torch.allclose(got0, want0, atol=1e-4, rtol=1e-4)


def test_autotune_jac_batches_cpu_returns_caps() -> None:
    model = _TinyLM()
    d, p = autotune_jac_batches(
        model, seq_len=12, max_dim_batch=16, max_prompt_batch=3, skip_first=2
    )
    assert d == 16
    assert p == 3
