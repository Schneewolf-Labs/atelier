"""
Smoke-test the QwenImageAdapter end-to-end with one prompt, one image,
one forward pass. Validates loading + shapes before committing to the
multi-hour flammen training run.

Wall time on a cold cache: ~15 min (mostly Qwen-Image weights download).
On a warm cache: ~1 min.
"""
import os
from pathlib import Path

import torch
from PIL import Image

from atelier.adapters import QwenImageAdapter

QWEN = os.environ.get("QWEN_IMAGE_PATH", "Qwen/Qwen-Image")
SAMPLE = Path(os.path.expanduser("~/flammen-lora-dataset/images")).glob("*.png")
sample_img_path = next(SAMPLE, None)
assert sample_img_path is not None, "no sample image found"

def vram(stage):
    if torch.cuda.is_available():
        a = torch.cuda.memory_allocated() / 1024**3
        r = torch.cuda.memory_reserved() / 1024**3
        print(f"  [VRAM @ {stage}] allocated={a:.1f} GiB reserved={r:.1f} GiB")


print(f"loading adapter from {QWEN} (defer_transformer=True) …")
adapter = QwenImageAdapter(QWEN)
vram("after adapter init (text encoder + VAE on GPU, transformer on CPU)")

print(f"\nencoding image {sample_img_path.name} …")
img = Image.open(sample_img_path).convert("RGB")
target_latents = adapter.encode_images([img], height=1024, width=1024)
print(f"  target_latents shape: {tuple(target_latents.shape)}  dtype={target_latents.dtype}")

print("\nencoding text 'a smoke test prompt' …")
text_out = adapter.encode_text(["a smoke test prompt"])
for k, v in text_out.items():
    if isinstance(v, torch.Tensor):
        print(f"  {k}: shape={tuple(v.shape)}  dtype={v.dtype}")
vram("after one encode_text + encode_images")

# Move target latents to CPU so freeing encoders doesn't drop them
target_latents_cpu = target_latents.detach().cpu()
text_out_cpu = {
    k: (v.detach().cpu() if isinstance(v, torch.Tensor) else v)
    for k, v in text_out.items()
}
del target_latents, text_out

print("\nfreeing encoders …")
adapter.free_encoders()
vram("after free_encoders (transformer still on CPU)")

print("\nmoving transformer to GPU …")
adapter.move_transformer_to_device()
vram("after move_transformer_to_device")

print("\nbuilding fake batch + running one forward pass …")
target_perm = target_latents_cpu.permute(0, 2, 1, 3, 4).to(adapter._device, adapter._dtype)
noise = torch.randn_like(target_perm)
sigmas = torch.tensor([0.5], device=target_perm.device).view(1, 1, 1, 1, 1)
noisy = (1 - sigmas) * target_perm + sigmas * noise
timesteps = torch.tensor([500], device=target_perm.device)
batch = {
    "prompt_embeds": text_out_cpu["prompt_embeds"].to(adapter._device, adapter._dtype),
    "prompt_embeds_mask": text_out_cpu.get("prompt_embeds_mask").to(adapter._device)
        if text_out_cpu.get("prompt_embeds_mask") is not None else None,
}

with torch.no_grad():
    pred = adapter.forward(adapter.model, noisy, timesteps, batch)
print(f"  prediction shape: {tuple(pred.shape)}  dtype={pred.dtype}")
vram("after forward pass")

# Sanity: prediction should match target shape after unpacking
expected_h = noisy.shape[3] * adapter._vae_scale_factor
expected_w = noisy.shape[4] * adapter._vae_scale_factor
print(f"  expected H×W (post-unpack): {expected_h}×{expected_w}")
assert pred.shape[-2:] == (expected_h, expected_w), \
    f"shape mismatch: got {pred.shape}, expected (..., {expected_h}, {expected_w})"
print("\nSMOKE TEST PASSED ✓")
