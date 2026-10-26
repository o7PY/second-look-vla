"""R4: arm mask and static mask are independent. Static-only mode: a static patch is excluded, an arm patch is not."""
import numpy as np
import pytest
import torch

from second_look.model import DeltaDINO


def _static_pt(m):
    return (3, 4)


@pytest.fixture(scope="module")
def models():
    out = {}
    for mode, use_static in (("static", True), ("feature", True), ("none", True), ("feature", False)):
        m = DeltaDINO(lora_rank=0, mask_mode=mode, use_static=use_static).eval()
        if use_static:
            m.static[3, 4] = True
        out[(mode, use_static)] = m
    return out


def _out(model, feats_fn, arm):
    B = arm.shape[0]
    f = feats_fn()
    model.features = lambda x: f.clone()
    img = torch.zeros(B, 3, 224, 224)
    return model(img, img, model.apply_static(arm))


def test_apply_static_modes(models):
    arm = torch.zeros(2, 16, 16, dtype=torch.bool); arm[:, 8, 8] = True
    s = models[("static", True)].apply_static(arm)
    assert s[0, 3, 4] and not s[0, 8, 8] and int(s[0].sum()) == 1                # static only: arm ignored
    f = models[("feature", True)].apply_static(arm)
    assert f[0, 3, 4] and f[0, 8, 8] and int(f[0].sum()) == 2                    # arm OR static
    assert torch.equal(models[("feature", False)].apply_static(arm), arm)       # no static mask: arm unchanged
    assert torch.equal(models[("static", True)].apply_static(s), s)             # idempotent
    d = models[("static", True)].display_mask(arm)
    assert d[0, 3, 4] and d[0, 8, 8]                                             # strata / pointing always exclude arm and static


def test_static_only_excludes_static_patch_but_not_arm_patch(models):
    torch.manual_seed(0)
    B = 2
    base = torch.randn(2 * B, 384, 16, 16)
    arm = torch.zeros(B, 16, 16, dtype=torch.bool); arm[:, 8, 8] = True
    m = models[("static", True)]
    s0, _ = _out(m, lambda: base, arm)

    def perturbed(patch):
        f = base.clone()
        for half in (0, B):
            f[half:half + B, :, patch[0], patch[1]] = torch.randn(B, 384) * 6
        return f
    s_static, _ = _out(m, lambda: perturbed((3, 4)), arm)                        # change features inside the static patch
    s_arm, _ = _out(m, lambda: perturbed((8, 8)), arm)                           # change features inside the (rendered) arm patch
    assert torch.allclose(s0, s_static, atol=1e-5), "static patch must not influence the output"
    assert not torch.allclose(s0, s_arm, atol=1e-3), "arm patch must still influence the output in static-only mode"


def test_feature_mode_excludes_both(models):
    torch.manual_seed(1)
    B = 2
    base = torch.randn(2 * B, 384, 16, 16)
    arm = torch.zeros(B, 16, 16, dtype=torch.bool); arm[:, 8, 8] = True
    m = models[("feature", True)]
    s0, _ = _out(m, lambda: base, arm)
    for patch in ((3, 4), (8, 8)):
        f = base.clone()
        for half in (0, B):
            f[half:half + B, :, patch[0], patch[1]] = torch.randn(B, 384) * 6
        s1, _ = _out(m, lambda: f, arm)
        assert torch.allclose(s0, s1, atol=1e-5), patch


def test_static_mode_requires_static_buffer_and_roundtrips():
    with pytest.raises(AssertionError):
        DeltaDINO(lora_rank=0, mask_mode="static", use_static=False)
    a = DeltaDINO(lora_rank=0, mask_mode="static", use_static=True); a.static[0, 11] = True
    b = DeltaDINO(lora_rank=0, mask_mode="static", use_static=True); b.load_trainable(a.trainable_state())
    assert torch.equal(a.static, b.static)
