"""Deterministic tiny Qwen-Image 2.1 DiT cases, shared by the tests and the sd.cpp parity script.

Weights and inputs come from a closed-form hash (no RNG), so they are identical
across torch versions and machines. ``tests/data/qwen_image_2_1_sdcpp.json``
holds stable-diffusion.cpp's outputs for these cases (regenerate with
``scripts/sdcpp_parity/dit_parity.py --write-golden``).
"""

import math

import torch

from atelier.models.qwen_image_2_1 import QwenImage21Config, QwenImage21Transformer

CASES = {
    "t2i": dict(fused=False, token_types=[0] * 7, refs=[], target=(4, 6), t=700.0),
    "t2i_odd": dict(fused=True, token_types=[0] * 5, refs=[], target=(5, 3), t=999.0),
    "edit_two_refs": dict(fused=True, token_types=[0, 1, 1, 0, 2, 2, 2, 2, 2, 2, 0, 0],
                          refs=[(2, 4), (4, 6)], target=(6, 6), t=100.0),
    "edit_ref_first": dict(fused=False, token_types=[1, 1, 0, 0, 0], refs=[(2, 4)], target=(4, 4), t=500.0),
}

CHANNELS = 8
CONTEXT_DIM = 32


def hashed(shape, seed, scale=1.0):
    """Pseudo-random values in [-scale, scale) from the classic fract(sin(x) * 43758.5453) hash."""
    n = math.prod(shape)
    i = torch.arange(n, dtype=torch.float64)
    u = torch.frac(torch.abs(torch.sin(i * 12.9898 + seed * 78.233) * 43758.5453))
    return ((u * 2 - 1) * scale).float().reshape(shape)


def build_case(name):
    """→ (model, x, timestep, context, token_types, refs) for one of CASES."""
    case = CASES[name]
    cfg = QwenImage21Config(in_channels=CHANNELS, out_channels=CHANNELS, hidden_size=256, context_dim=CONTEXT_DIM,
                            head_dim=128, intermediate_size=64, num_layers=2, fused_mlp=case["fused"])
    model = QwenImage21Transformer(cfg).eval()
    with torch.no_grad():
        for seed, (_, p) in enumerate(sorted(model.named_parameters())):
            p.copy_(hashed(p.shape, seed, scale=0.14))
    H, W = case["target"]
    L = len(case["token_types"])
    x = hashed((1, CHANNELS, H, W), 1001)
    context = hashed((1, L, CONTEXT_DIM), 1002)
    refs = [hashed((1, CHANNELS, h, w), 1003 + i) for i, (h, w) in enumerate(case["refs"])]
    return model, x, torch.tensor([case["t"]]), context, torch.tensor(case["token_types"]), refs
