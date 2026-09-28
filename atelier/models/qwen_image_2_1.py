"""Native PyTorch port of the Qwen-Image 2.1 diffusion transformer.

diffusers has no Qwen-Image 2.1 support, so this is a direct port of
stable-diffusion.cpp's ``src/model/diffusion/qwen_image_2_1.hpp``, with
parameter names matching the released (Comfy-Org) checkpoint so its
state dict loads as-is.

Architecture, briefly — it is *not* the Qwen-Image MMDiT:

- **One stream.** Text tokens, reference-image tokens and the noisy target
  share a single sequence and a single set of weights. Reference images
  replace their ``<|image_pad|>`` slot tokens inside the prompt; the target
  goes last.
- **Block-causal attention.** A text query sees every key up to and
  including itself; an image query sees every key up to the end of its own
  image. So text is causal, images are bidirectional, and the target sees
  everything. The prefix (text + references) never sees the target.
- **Two modulation rows.** One global ``modulation`` MLP turns the
  timestep embedding into scale/gate vectors; the target is modulated with
  the embedding of ``t`` and the prefix with the embedding of ``t = 0``
  (it is clean conditioning). Scale-only (``x * (1 + s)``, no shift);
  residual gates are ``tanh``-squashed.
- **3-axis RoPE** (16/56/56 dims, theta 10000, interleaved pairs). Text
  tokens get ``(p, p, p)`` with ``p`` counting up; each image gets
  ``(p, h - ceil(H/2), w - ceil(W/2))`` — a centred grid — and advances
  ``p`` by ``max(H, W)``.
- Latents are 64-channel at 16x (one token per latent pixel, no packing).
- Timestep input is ``sigma * 1000``; the output is the flow velocity
  ``noise - x0``.
"""

import math
import os
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint

TEXT = -1  # segment.image_index for text


@dataclass
class QwenImage21Config:
    in_channels: int = 64
    out_channels: int = 64
    hidden_size: int = 4096
    context_dim: int = 4096
    head_dim: int = 128
    intermediate_size: int = 12288
    num_layers: int = 32
    fused_mlp: bool = False
    axes_dim: tuple = (16, 56, 56)
    theta: float = 10000.0

    @property
    def num_heads(self):
        return self.hidden_size // self.head_dim

    @classmethod
    def from_state_dict(cls, sd):
        """Infer the config from checkpoint shapes, as sd.cpp does."""
        cfg = cls()
        if "img_in.weight" in sd:
            cfg.hidden_size, cfg.in_channels = sd["img_in.weight"].shape
        if "proj_out.weight" in sd:
            cfg.out_channels = sd["proj_out.weight"].shape[0]
        if "txt_in.in_layer.weight" in sd:
            cfg.context_dim = sd["txt_in.in_layer.weight"].shape[1]
        if "transformer_blocks.0.attn.norm_q.weight" in sd:
            cfg.head_dim = sd["transformer_blocks.0.attn.norm_q.weight"].shape[0]
        if "transformer_blocks.0.img_mlp.gate_up.weight" in sd:
            cfg.fused_mlp = True
            cfg.intermediate_size = sd["transformer_blocks.0.img_mlp.gate_up.weight"].shape[0] // 2
        elif "transformer_blocks.0.img_mlp.proj.weight" in sd:
            cfg.intermediate_size = sd["transformer_blocks.0.img_mlp.proj.weight"].shape[0]
        layers = {int(k.split(".")[1]) for k in sd if k.startswith("transformer_blocks.")}
        if layers:
            cfg.num_layers = max(layers) + 1
        return cfg


# ---------------------------------------------------------------------------
# Sequence layout
# ---------------------------------------------------------------------------


@dataclass
class Segment:
    start: int          # position in the joint sequence
    end: int
    context_start: int  # first row in the text context this segment covers
    image_index: int    # TEXT, or index into image_shapes


@dataclass
class Layout:
    segments: list = field(default_factory=list)
    positions: list = field(default_factory=list)  # [(axis0, axis1, axis2)] per token
    prefix_length: int = 0

    @property
    def length(self):
        return len(self.positions)


def build_layout(token_types, image_shapes):
    """Joint-sequence layout from per-token types and latent image shapes.

    Args:
        token_types: length-L sequence over the text context; ``0`` for text,
            ``i + 1`` for the slot tokens of reference image ``i``.
        image_shapes: ``[(H, W), ...]`` latent sizes — references in order,
            then the target last.

    Each reference must own exactly ``H * W / 4`` consecutive slot tokens
    (Qwen3-VL emits one token per 32 px; the latent has one per 16 px).
    """
    token_types = [int(t) for t in token_types]
    if not image_shapes:
        raise ValueError("image_shapes must at least contain the target")
    layout = Layout()
    position = 0
    next_image = 0

    def append_image(index, context_start):
        nonlocal position
        h, w = image_shapes[index]
        start = len(layout.positions)
        layout.segments.append(Segment(start, start + h * w, context_start, index))
        for y in range(h):
            for x in range(w):
                layout.positions.append((position, y - (h - h // 2), x - (w - w // 2)))
        position += max(h, w)

    i, n = 0, len(token_types)
    while i < n:
        tag, begin = token_types[i], i
        i += 1
        while i < n and token_types[i] == tag:
            i += 1
        if tag != 0:
            h, w = image_shapes[next_image] if next_image < len(image_shapes) else (0, 0)
            if tag != next_image + 1 or next_image + 1 >= len(image_shapes) or (i - begin) * 4 != h * w:
                raise ValueError("vision slots and reference latents must have matching sizes")
            append_image(next_image, begin)
            next_image += 1
        else:
            start = len(layout.positions)
            layout.segments.append(Segment(start, start + i - begin, begin, TEXT))
            for _ in range(begin, i):
                layout.positions.append((position, position, position))
                position += 1
    if next_image + 1 != len(image_shapes):
        raise ValueError("missing reference image slots")
    layout.prefix_length = len(layout.positions)
    append_image(next_image, n)
    return layout


def attention_mask(layout, device=None):
    """Boolean ``[L, L]`` mask (True = attend) for the block-causal pattern."""
    L = layout.length
    q = torch.arange(L, device=device)[:, None]
    k = torch.arange(L, device=device)[None, :]
    seg_end = torch.empty(L, dtype=torch.long, device=device)
    is_text = torch.zeros(L, dtype=torch.bool, device=device)
    for s in layout.segments:
        seg_end[s.start:s.end] = s.end
        is_text[s.start:s.end] = s.image_index == TEXT
    allowed = k < seg_end[:, None]
    return allowed & (~is_text[:, None] | (k <= q))


def rope_frequencies(positions, axes_dim, theta, device=None):
    """``[L, head_dim/2]`` complex rotations for 3-axis interleaved RoPE.

    Angles are computed in float64 on CPU (MPS has no float64), then moved.
    """
    pos = torch.as_tensor(positions, dtype=torch.float64)
    freqs = []
    for axis, dim in enumerate(axes_dim):
        omega = 1.0 / theta ** (torch.arange(0, dim, 2, dtype=torch.float64) / dim)
        freqs.append(pos[:, axis:axis + 1] * omega[None, :])
    angles = torch.cat(freqs, dim=-1)
    return torch.polar(torch.ones_like(angles), angles).to(torch.complex64).to(device)


def apply_rope(x, freqs):
    """x: ``[B, H, L, D]``; rotates adjacent pairs ``(x[2i], x[2i+1])``."""
    xc = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    return torch.view_as_real(xc * freqs).flatten(-2).to(x.dtype)


# ---------------------------------------------------------------------------
# Modules (names mirror the checkpoint)
# ---------------------------------------------------------------------------


class ZeroCenteredRMSNorm(nn.Module):
    """RMSNorm whose weight is stored as an offset from 1."""

    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x * (1.0 + self.weight.float())).to(dtype)


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x * self.weight.float()).to(dtype)


def timestep_embedding(t, dim=256, max_period=10000):
    """``[cos | sin]`` sinusoidal embedding (ggml / diffusers flip_sin_to_cos order)."""
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(half, dtype=torch.float32, device=t.device) / half)
    args = t.float()[:, None] * freqs[None]
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.linear_1 = nn.Linear(256, hidden_size, bias=False)
        self.linear_2 = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, x):
        return self.linear_2(F.silu(self.linear_1(x)))


class TimeTextEmbed(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.timestep_embedder = TimestepEmbedder(hidden_size)


class TextProjection(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.text_norm = ZeroCenteredRMSNorm(cfg.context_dim)
        self.in_layer = nn.Linear(cfg.context_dim, cfg.hidden_size, bias=False)
        self.out_layer = nn.Linear(cfg.hidden_size, cfg.hidden_size, bias=False)

    def forward(self, x):
        # ggml_gelu is the tanh approximation
        return self.out_layer(F.gelu(self.in_layer(self.text_norm(x)), approximate="tanh"))


class Attention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        d, hd = cfg.hidden_size, cfg.head_dim
        self.heads = cfg.num_heads
        self.to_q = nn.Linear(d, d, bias=False)
        self.to_k = nn.Linear(d, d, bias=False)
        self.to_v = nn.Linear(d, d, bias=False)
        self.norm_q = RMSNorm(hd)
        self.norm_k = RMSNorm(hd)
        self.to_out = nn.ModuleList([nn.Linear(d, d, bias=False), nn.Dropout(0.0)])

    def forward(self, x, freqs, mask):
        b, L, _ = x.shape

        def heads(t):
            return t.view(b, L, self.heads, -1).transpose(1, 2)

        q = apply_rope(self.norm_q(heads(self.to_q(x))), freqs)
        k = apply_rope(self.norm_k(heads(self.to_k(x))), freqs)
        v = heads(self.to_v(x))
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        return self.to_out[0](out.transpose(1, 2).reshape(b, L, -1))


class MLP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.fused = cfg.fused_mlp
        if self.fused:
            self.gate_up = nn.Linear(cfg.hidden_size, 2 * cfg.intermediate_size, bias=False)
        else:
            self.proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
            self.gate_layer = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.out = nn.Linear(cfg.intermediate_size, cfg.hidden_size, bias=False)

    def forward(self, x):
        if self.fused:
            gate, up = self.gate_up(x).chunk(2, dim=-1)
        else:
            gate, up = self.gate_layer(x), self.proj(x)
        return self.out(up * F.silu(gate))


def modulate(x, rows, prefix_length, gate=False):
    """Row 0 (``t``) modulates the target, row 1 (``t = 0``) the prefix.

    rows: ``[B, 2, D]``. Scale form ``x * (1 + s)``; gate form ``x * tanh(g)``.
    """
    rows = torch.tanh(rows) if gate else 1.0 + rows
    rows = rows.to(x.dtype)
    target = x[:, prefix_length:] * rows[:, :1]
    if prefix_length == 0:
        return target
    return torch.cat([x[:, :prefix_length] * rows[:, 1:2], target], dim=1)


class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.img_norm1 = nn.LayerNorm(cfg.hidden_size, eps=1e-6, elementwise_affine=False)
        self.img_norm2 = nn.LayerNorm(cfg.hidden_size, eps=1e-6, elementwise_affine=False)
        self.attn = Attention(cfg)
        self.img_mlp = MLP(cfg)

    def forward(self, x, mod, freqs, mask, prefix_length):
        h = modulate(self.img_norm1(x), mod[0], prefix_length)
        x = x + modulate(self.attn(h, freqs, mask), mod[1], prefix_length, gate=True)
        h = modulate(self.img_norm2(x), mod[2], prefix_length)
        return x + modulate(self.img_mlp(h), mod[3], prefix_length, gate=True)


class NormOut(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.linear = nn.Linear(hidden_size, hidden_size, bias=False)
        self.norm = nn.LayerNorm(hidden_size, eps=1e-6, elementwise_affine=False)


class QwenImage21Transformer(nn.Module):
    """The Qwen-Image 2.1 DiT. ``state_dict()`` keys match the released checkpoint."""

    def __init__(self, config=None):
        super().__init__()
        cfg = config or QwenImage21Config()
        self.config = cfg
        self.time_text_embed = TimeTextEmbed(cfg.hidden_size)
        self.txt_in = TextProjection(cfg)
        self.img_in = nn.Linear(cfg.in_channels, cfg.hidden_size, bias=False)
        self.modulation = nn.Sequential(nn.SiLU(), nn.Linear(cfg.hidden_size, 4 * cfg.hidden_size, bias=False))
        self.transformer_blocks = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.num_layers)])
        self.norm_out = NormOut(cfg.hidden_size)
        self.proj_out = nn.Linear(cfg.hidden_size, cfg.out_channels, bias=False)
        self.gradient_checkpointing = False

    # -- trainer hooks ------------------------------------------------------
    @property
    def device(self):
        return self.img_in.weight.device

    @property
    def dtype(self):
        return self.img_in.weight.dtype

    def enable_gradient_checkpointing(self):
        self.gradient_checkpointing = True

    def disable_gradient_checkpointing(self):
        self.gradient_checkpointing = False

    # -- forward ------------------------------------------------------------
    def forward(self, x, timestep, context, token_types=None, ref_latents=()):
        """Single-sample forward (the layout is per-sample).

        Args:
            x: ``[1, C, H, W]`` noisy target latent.
            timestep: ``[1]`` in ``[0, 1000]`` (``sigma * 1000``).
            context: ``[1, L, context_dim]`` Qwen3-VL hidden states.
            token_types: ``[L]`` ints (0 text, i+1 image-i slot) or None.
            ref_latents: reference latents ``[1, C, H_i, W_i]``, in slot order.

        Returns the ``[1, C_out, H, W]`` velocity prediction.
        """
        if x.shape[0] != 1:
            raise ValueError("QwenImage21Transformer.forward is per-sample; loop over the batch")
        cfg = self.config
        device, dtype = x.device, x.dtype
        L = context.shape[1]
        if token_types is None:
            token_types = torch.zeros(L, dtype=torch.long)
        shapes = [tuple(r.shape[-2:]) for r in ref_latents] + [tuple(x.shape[-2:])]
        layout = build_layout(token_types.tolist(), shapes)

        # t and t = 0 → two embedding rows per sample
        t = torch.cat([timestep.reshape(1).float(), torch.zeros(1, device=device)])
        temb = self.time_text_embed.timestep_embedder(timestep_embedding(t).to(dtype))  # [2, D]
        mod = self.modulation(temb).chunk(4, dim=-1)                                     # 4 x [2, D]
        mod = [m.unsqueeze(0) for m in mod]                                               # 4 x [1, 2, D]

        text = self.txt_in(context.to(dtype))
        images = list(ref_latents) + [x]
        pieces = []
        for seg in layout.segments:
            if seg.image_index == TEXT:
                pieces.append(text[:, seg.context_start:seg.context_start + seg.end - seg.start])
            else:
                img = images[seg.image_index].to(dtype)
                pieces.append(self.img_in(img.flatten(2).transpose(1, 2)))
        h = torch.cat(pieces, dim=1)

        freqs = rope_frequencies(layout.positions, cfg.axes_dim, cfg.theta, device=device)
        mask = attention_mask(layout, device=device)
        for block in self.transformer_blocks:
            if self.gradient_checkpointing and torch.is_grad_enabled():
                h = torch.utils.checkpoint.checkpoint(
                    block, h, mod, freqs, mask, layout.prefix_length, use_reentrant=False,
                )
            else:
                h = block(h, mod, freqs, mask, layout.prefix_length)

        h = h[:, layout.prefix_length:]
        scale = self.norm_out.linear(F.silu(temb[:1]))                                   # t row only
        h = self.norm_out.norm(h) * (1.0 + scale[:, None].to(h.dtype))
        out = self.proj_out(h)                                                           # [1, HW, C]
        _, _, H, W = x.shape
        return out.transpose(1, 2).reshape(1, -1, H, W)


# ---------------------------------------------------------------------------
# VAE
# ---------------------------------------------------------------------------
#
# The Qwen-Image 2.1 VAE is a Wan2.2-style residual VAE (AvgDown3D / DupUp3D
# shortcuts) with a 64-channel latent, RGBA input and one more downsampling
# level (16x). diffusers' AutoencoderKLWan implements exactly those blocks, so
# it is configured here rather than re-ported; only the checkpoint key names
# and 2D-vs-3D conv kernels need converting. Constants from sd.cpp
# (src/model/vae/wan_vae.hpp, VERSION_QWEN_IMAGE_2_1).

VAE_LATENTS_MEAN = [
    0.5126, 0.7721, -0.0631, 1.3506, -0.7855, -2.1025, -0.3458, 1.3722,
    1.8873, -1.7177, -0.6510, 0.2732, 0.7562, -0.6163, -1.0277, 3.8363,
    2.0210, 0.0472, 0.9320, 2.0087, 2.4954, -0.1391, -1.4249, 1.8464,
    -0.5236, 1.2826, 3.7046, -1.3035, 2.7286, -1.4518, -1.9036, -1.9955,
    -0.0342, -1.0265, -0.7636, 3.0555, 0.0746, -3.0751, -0.1076, 1.7376,
    -1.0914, -1.9435, -0.2784, -1.3680, 0.4809, -0.4433, 0.3764, 0.5729,
    -2.0595, 1.0960, -1.3260, -2.0211, -5.0179, 0.5275, 4.0162, 1.8505,
    0.3026, 1.9373, 1.4937, 0.2632, 0.5547, -1.7121, -0.1562, 0.0304,
]
VAE_LATENTS_STD = [
    3.2001, 3.2936, 3.4321, 3.0091, 3.1061, 4.0379, 4.0705, 3.7910,
    3.0785, 3.6500, 3.9308, 3.0904, 2.8778, 3.7675, 3.7320, 5.0756,
    3.2864, 4.0397, 3.1317, 4.0443, 2.9249, 3.9454, 3.0988, 4.2489,
    3.4896, 3.8513, 3.9323, 3.4719, 3.7498, 4.2830, 3.5694, 4.2467,
    3.9037, 3.2947, 5.0770, 3.5075, 3.2700, 3.4767, 2.8063, 5.1125,
    3.5327, 4.7833, 3.1286, 4.1819, 3.8527, 3.8312, 3.5605, 4.3875,
    3.9624, 4.0168, 3.5643, 4.0550, 5.5614, 4.2963, 4.4080, 3.4959,
    3.8747, 3.7608, 3.5735, 3.1490, 3.7662, 3.6746, 3.4563, 3.8161,
]

VAE_CONFIG = dict(
    base_dim=96,
    decoder_base_dim=144,
    z_dim=64,
    dim_mult=[1, 2, 4, 8, 8],
    num_res_blocks=2,
    attn_scales=[],
    temperal_downsample=[False, True, True, True],
    is_residual=True,
    in_channels=4,
    out_channels=4,
    patch_size=None,
    scale_factor_spatial=16,
    latents_mean=VAE_LATENTS_MEAN,
    latents_std=VAE_LATENTS_STD,
)

_RESNET_PARTS = {"0": "norm1", "2": "conv1", "3": "norm2", "6": "conv2"}


def convert_vae_key(key):
    """Original (Wan / Comfy) Qwen-Image 2.1 VAE key → diffusers AutoencoderKLWan key.

    Inverse of sd.cpp's ``convert_diffusers_to_original_wan_vae(..., qwen_image_2_1=True)``.
    Keys already in diffusers form pass through unchanged.
    """
    import re

    for prefix in ("first_stage_model.", "vae."):
        if key.startswith(prefix):
            key = key[len(prefix):]
    if key.startswith("conv1."):
        return "quant_conv." + key[len("conv1."):]
    if key.startswith("conv2."):
        return "post_quant_conv." + key[len("conv2."):]

    def resnet(prefix, rest):
        m = re.match(r"(shortcut|residual\.(\d+))\.(.*)", rest)
        if m is None:
            return None
        if m.group(1) == "shortcut":
            return f"{prefix}.conv_shortcut.{m.group(3)}"
        return f"{prefix}.{_RESNET_PARTS[m.group(2)]}.{m.group(3)}"

    m = re.match(r"(encoder|decoder)\.(downsamples|upsamples)\.(\d+)\.(downsamples|upsamples)\.(\d+)\.(.*)", key)
    if m:
        side, _, i, _, j, rest = m.groups()
        blocks = "down_blocks" if side == "encoder" else "up_blocks"
        sampler_index = "2" if side == "encoder" else "3"
        if j == sampler_index:
            return f"{side}.{blocks}.{i}.{'downsampler' if side == 'encoder' else 'upsampler'}.{rest}"
        return resnet(f"{side}.{blocks}.{i}.resnets.{j}", rest) or key

    m = re.match(r"(encoder|decoder)\.middle\.(\d)\.(.*)", key)
    if m:
        side, idx, rest = m.groups()
        if idx == "1":
            return f"{side}.mid_block.attentions.0.{rest}"
        return resnet(f"{side}.mid_block.resnets.{0 if idx == '0' else 1}", rest) or key

    m = re.match(r"(encoder|decoder)\.(conv1|head\.0|head\.2)\.(.*)", key)
    if m:
        side, part, rest = m.groups()
        return f"{side}.{ {'conv1': 'conv_in', 'head.0': 'norm_out', 'head.2': 'conv_out'}[part] }.{rest}"
    return key


def _fit_tensor(value, target_shape):
    """Reshape / embed a checkpoint tensor into the diffusers parameter shape.

    Image-only exports store the causal 3D convs as 2D kernels (or 3D with a
    singleton temporal kernel). On a single frame a causal conv only sees its
    last temporal tap (the others hit the zero padding), so those kernels go
    into slot ``kt - 1``.
    """
    target_shape = tuple(target_shape)
    if tuple(value.shape) == target_shape:
        return value
    if value.numel() == math.prod(target_shape) and len(target_shape) != 5:
        return value.reshape(target_shape)
    if len(target_shape) == 5:
        if value.ndim == 4:
            value = value.unsqueeze(2)
        if value.ndim == 5 and value.shape[2] == 1 and value.shape[:2] == target_shape[:2]:
            out = value.new_zeros(target_shape)
            out[:, :, -1] = value[:, :, 0]
            return out
    if value.numel() == math.prod(target_shape):
        return value.reshape(target_shape)
    raise ValueError(f"cannot fit tensor of shape {tuple(value.shape)} into {target_shape}")


def convert_vae_state_dict(state_dict, reference_state_dict):
    """Rename + reshape an original-format VAE state dict to fit ``reference_state_dict``."""
    out = {}
    for key, value in state_dict.items():
        new_key = convert_vae_key(key)
        if new_key not in reference_state_dict:
            raise KeyError(f"unexpected VAE key {key!r} (→ {new_key!r})")
        out[new_key] = _fit_tensor(value, reference_state_dict[new_key].shape)
    missing = set(reference_state_dict) - set(out)
    if missing:
        raise KeyError(f"VAE checkpoint is missing {len(missing)} keys, e.g. {sorted(missing)[:3]}")
    return out


def build_vae(state_dict=None, **overrides):
    """An ``AutoencoderKLWan`` configured as the Qwen-Image 2.1 VAE, optionally loaded."""
    from diffusers import AutoencoderKLWan

    vae = AutoencoderKLWan(**{**VAE_CONFIG, **overrides})
    if state_dict is not None:
        vae.load_state_dict(convert_vae_state_dict(state_dict, vae.state_dict()))
    return vae


# ---------------------------------------------------------------------------
# Text / vision conditioning (Qwen3-VL-8B-Instruct)
# ---------------------------------------------------------------------------
#
# From sd.cpp (src/conditioning/conditioner.hpp, VERSION_QWEN_IMAGE_2_1):
#   - fixed system prompt, stripped from the returned hidden states
#   - references as "<imageN><|vision_start|>{pads}<|vision_end|>", space-separated,
#     before the user text; an empty prompt becomes " "
#   - hidden states of the *last decoder layer before the final norm*
#   - the vision tower sees the same pixels as the VAE, alpha composited on white

SYSTEM_PROMPT = "<|im_start|>system\nComprehend and analyze the provided prompt.<|im_end|>\n"
IMAGE_PAD = "<|image_pad|>"


def build_prompt(text, num_images=0):
    """The full chat string (system prefix included), one image pad per reference."""
    refs = " ".join(f"<image{i + 1}><|vision_start|>{IMAGE_PAD}<|vision_end|>" for i in range(num_images))
    return SYSTEM_PROMPT + "<|im_start|>user\n" + refs + (text or " ") + "<|im_end|>\n<|im_start|>assistant\n"


def token_types_from_ids(input_ids, image_pad_id):
    """0 for text, ``i + 1`` for every pad token of the i-th contiguous image run."""
    types = torch.zeros(len(input_ids), dtype=torch.long)
    image, prev = 0, False
    for i, tok in enumerate(input_ids.tolist() if torch.is_tensor(input_ids) else input_ids):
        is_pad = tok == image_pad_id
        if is_pad and not prev:
            image += 1
        if is_pad:
            types[i] = image
        prev = is_pad
    return types


def composite_on_white(image):
    """RGBA/LA/P → RGB over white, as sd.cpp feeds the vision tower."""
    from PIL import Image

    if image.mode in ("RGBA", "LA") or (image.mode == "P" and "transparency" in image.info):
        rgba = image.convert("RGBA")
        background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        return Image.alpha_composite(background, rgba).convert("RGB")
    return image.convert("RGB")


class Qwen3VLConditioner:
    """Encodes prompts (+ optional reference images) into DiT context.

    Returns ``(hidden [L, D], token_types [L])`` per prompt; ``token_types``
    marks which context rows are vision slots for which reference.
    """

    def __init__(self, processor, model, device="cuda"):
        self.processor = processor
        self.model = model.to(device).eval().requires_grad_(False)
        self.device = device
        tok = processor.tokenizer
        self.image_pad_id = tok.convert_tokens_to_ids(IMAGE_PAD)
        self.prefix_length = len(tok(SYSTEM_PROMPT, add_special_tokens=False).input_ids)

    @classmethod
    def from_pretrained(cls, pretrained_path, device="cuda", dtype=torch.bfloat16):
        """Load Qwen3-VL-8B-Instruct (the stock HF release is the 2.1 text encoder)."""
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

        processor = AutoProcessor.from_pretrained(pretrained_path)
        model = Qwen3VLForConditionalGeneration.from_pretrained(pretrained_path, torch_dtype=dtype)
        return cls(processor, model, device=device)

    def to(self, device):
        self.model.to(device)
        self.device = device
        return self

    def _last_layer(self):
        return self.model.model.language_model.layers[-1]

    @torch.no_grad()
    def encode(self, text, images=None):
        images = [composite_on_white(im) for im in (images or [])]
        for im in images:
            if im.width % 32 or im.height % 32:
                raise ValueError(f"reference images must be multiples of 32 px, got {im.size}")
        inputs = self.processor(
            text=[build_prompt(text, len(images))],
            images=images or None,
            return_tensors="pt",
            # sizes are already multiples of 32 — keep the VLM grid aligned with the VAE latent
            do_resize=False,
        ).to(self.device)

        captured = {}

        def hook(_module, _args, output):
            captured["h"] = output[0] if isinstance(output, tuple) else output

        handle = self._last_layer().register_forward_hook(hook)
        try:
            self.model.model(**inputs, use_cache=False)
        finally:
            handle.remove()

        ids = inputs["input_ids"][0]
        hidden = captured["h"][0, self.prefix_length:]
        types = token_types_from_ids(ids[self.prefix_length:].cpu(), self.image_pad_id)
        return hidden, types


# ---------------------------------------------------------------------------
# Checkpoint I/O
# ---------------------------------------------------------------------------

DIT_PREFIXES = ("model.diffusion_model.", "diffusion_model.", "transformer.")


def strip_prefix(state_dict, prefixes):
    for prefix in prefixes:
        if any(k.startswith(prefix) for k in state_dict):
            return {k[len(prefix):] if k.startswith(prefix) else k: v for k, v in state_dict.items()}
    return state_dict


def _check_unquantized(state_dict, what):
    quantized = [k for k, v in state_dict.items()
                 if k.endswith(".weight_scale") or v.dtype in (torch.int8, torch.uint8)
                 or "float8" in str(v.dtype)]
    if quantized:
        raise ValueError(
            f"{what} checkpoint is quantized (e.g. {quantized[0]!r}); training needs the bf16/fp16/fp32 "
            "weights — the int8-convrot / fp8 releases are inference-only."
        )


def load_transformer(path, dtype=torch.bfloat16, device="cpu"):
    """Build a QwenImage21Transformer from a single-file checkpoint (config inferred from shapes)."""
    from safetensors.torch import load_file

    sd = strip_prefix(load_file(os.path.expanduser(str(path))), DIT_PREFIXES)
    _check_unquantized(sd, "Qwen-Image 2.1 transformer")
    with torch.device("meta"):
        model = QwenImage21Transformer(QwenImage21Config.from_state_dict(sd))
    model.load_state_dict({k: v.to(dtype) for k, v in sd.items()}, strict=True, assign=True)
    return model.to(device)


def load_vae(path, dtype=torch.float32, device="cpu"):
    """Build the VAE from a single-file checkpoint (original / Comfy or diffusers key names)."""
    from safetensors.torch import load_file

    sd = load_file(os.path.expanduser(str(path)))
    _check_unquantized(sd, "Qwen-Image 2.1 VAE")
    return build_vae({k: v.float() for k, v in sd.items()}).to(device=device, dtype=dtype)
