"""Compare atelier's Qwen-Image 2.1 VAE (diffusers AutoencoderKLWan) against sd.cpp's WanVAERunner.

    python scripts/sdcpp_parity/vae_parity.py --harness ./harness [--kernels 2d|3d]

Builds a random full-size checkpoint in the original (Wan / Comfy) key layout —
``2d`` = image-export conv kernels, ``3d`` = causal 3D kernels — loads it into
sd.cpp as-is and into atelier through convert_vae_state_dict, then compares
encode (raw mean) and decode. sd.cpp runs its convs in fp16, so expect ~5e-3
relative; structural mistakes show up as O(1).
"""
import argparse
import subprocess
import sys
import tempfile

import numpy as np
import torch
from safetensors.torch import save_file

from atelier.models.qwen_image_2_1 import build_vae

ap = argparse.ArgumentParser()
ap.add_argument("--harness", default="./harness")
ap.add_argument("--kernels", choices=["2d", "3d"], default="2d")
args = ap.parse_args()
D = tempfile.mkdtemp()


def to_original(name):
    """Python port of sd.cpp convert_diffusers_to_original_wan_vae(..., qwen_image_2_1=True)."""
    for i in range(5):
        idx = str(i)
        for side in ("encoder", "decoder"):
            enc = side == "encoder"
            old = side + (".down_blocks." if enc else ".up_blocks.") + idx + "."
            new = side + (".downsamples." if enc else ".upsamples.") + idx + "."
            if not name.startswith(old):
                continue
            name = new + name[len(old):]
            layers = "downsamples." if enc else "upsamples."
            for j in range(2 if enc else 3):
                orn, nrn = new + "resnets." + str(j) + ".", new + layers + str(j) + "."
                if name.startswith(orn + "conv_shortcut."):
                    name = nrn + "shortcut." + name[len(orn + "conv_shortcut."):]
                elif name.startswith(orn):
                    name = nrn + "residual." + name[len(orn):]
            sp = new + ("downsampler." if enc else "upsampler.")
            if name.startswith(sp):
                name = new + layers + ("2." if enc else "3.") + name[len(sp):]
    for a, b in [(".conv_in.", ".conv1."), (".norm_out.", ".head.0."), (".conv_out.", ".head.2."),
                 (".mid_block.attentions.0.", ".middle.1."), (".mid_block.resnets.0.", ".middle.0.residual."),
                 (".mid_block.resnets.1.", ".middle.2.residual.")]:
        name = name.replace(a, b)
    for a, b in [("quant_conv.", "conv1."), ("post_quant_conv.", "conv2.")]:
        if name.startswith(a):
            name = b + name[len(a):]
    if ".residual." in name:
        for a, b in [(".norm1.", ".0."), (".conv1.", ".2."), (".norm2.", ".3."), (".conv2.", ".6.")]:
            name = name.replace(a, b)
    return name


torch.manual_seed(0)
ref_model = build_vae()
original = {}
for k, v in ref_model.state_dict().items():
    shape = list(v.shape)
    if v.ndim == 5 and args.kernels == "2d":  # causal 3D conv → image-export 2D kernel
        shape = shape[:2] + shape[3:]
    if k.endswith("gamma"):
        t = 1.0 + 0.1 * torch.randn(shape)
    elif k.endswith("bias"):
        t = 0.02 * torch.randn(shape)
    else:
        fan_in = int(np.prod(shape[1:]))
        t = torch.randn(shape) / fan_in ** 0.5
    original["first_stage_model." + to_original(k)] = t.contiguous()
save_file(original, f"{D}/vae.safetensors")

vae = build_vae({k[len("first_stage_model."):]: v for k, v in original.items()}).eval()

H = W = 64
x = torch.rand(1, 4, H, W) * 2 - 1
x.numpy().tofile(f"{D}/img.bin")


def harness(*cli):
    r = subprocess.run([args.harness, "vae", f"{D}/vae.safetensors", *cli], capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stderr[-3000:])
        raise SystemExit("harness failed")


def report(name, ours, ref):
    err = (ours - ref).abs().max().item()
    scale = ref.abs().max().item()
    print(f"{name:10s} max|diff|={err:.2e}  max|ref|={scale:.3f}  rel={err / scale:.2e}")
    return err / scale


harness(f"{D}/img.bin", str(W), str(H), "4", f"{D}/lat.bin")
ref_lat = torch.from_numpy(np.fromfile(f"{D}/lat.bin", dtype=np.float32)).reshape(1, 64, H // 16, W // 16)
with torch.no_grad():
    ours_lat = vae.encode(x.unsqueeze(2)).latent_dist.mode().squeeze(2)
e1 = report("encode", ours_lat, ref_lat)

z = torch.randn(1, 64, 4, 4)
z.numpy().tofile(f"{D}/z.bin")
harness(f"{D}/z.bin", "4", "4", "64", f"{D}/dec.bin", "decode")
ref_img = torch.from_numpy(np.fromfile(f"{D}/dec.bin", dtype=np.float32)).reshape(1, 4, 64, 64)
with torch.no_grad():
    ours_img = vae.decode(z.unsqueeze(2)).sample.squeeze(2)
e2 = report("decode", ours_img.clamp(-1, 1), ref_img.clamp(-1, 1))
sys.exit(0 if max(e1, e2) < 2e-2 else 1)
