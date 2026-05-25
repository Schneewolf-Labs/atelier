"""
A/B evaluation: render the same prompt set with + without the trained
flammen-aesthetic-v1 LoRA. Saves all 12 images to ~/flammen-lora-output/eval/
plus a small index.html for side-by-side viewing.

The prompt set is novel (not in the training corpus) and shaped to test:
- distribution baseline (typical flame card prompt shape)
- different gender
- dark skin (validates the VLM-caption-grounding decision)
- fantasy character (validates non-human handling)
- out-of-distribution (catches catastrophic forgetting)
- natural-language prompt (validates the dual-caption training)
"""
import os
import time
from pathlib import Path

import torch
from diffusers import QwenImagePipeline

BASE = os.environ.get("QWEN_IMAGE_PATH", "Qwen/Qwen-Image")
LORA_DIR = Path(os.path.expanduser("~/flammen-lora-output/flammen-aesthetic-v1"))
OUT_DIR  = Path(os.path.expanduser("~/flammen-lora-output/eval"))
OUT_DIR.mkdir(parents=True, exist_ok=True)

PROMPTS = [
    ("1_distribution_baseline",
     "anime style, masterpiece, best quality, high detail, solo, looking at viewer, "
     "simple background, upper body, 1girl, blonde_hair, long_hair, blue_eyes, "
     "hoodie, gentle_smile"),
    ("2_male_baseline",
     "anime style, masterpiece, best quality, high detail, solo, looking at viewer, "
     "simple background, upper body, 1boy, brown_hair, short_hair, brown_eyes, "
     "glasses, casual_button_up_shirt, calm_expression"),
    ("3_dark_skin",
     "anime style, masterpiece, best quality, high detail, solo, looking at viewer, "
     "simple background, upper body, 1girl, dark_skin, black_hair, braids, brown_eyes, "
     "denim_jacket, confident_expression"),
    ("4_fantasy_character",
     "anime style, masterpiece, best quality, high detail, solo, looking at viewer, "
     "simple background, upper body, 1boy, dragon_horns, silver_hair, ahoge, "
     "heterochromia, slit_pupils, pale_skin"),
    ("5_out_of_distribution",
     "a serene landscape painting of misty mountains at sunset, no people, "
     "watercolor style"),
    ("6_natural_language",
     "A young woman with short brown hair, gentle smile, reading a book in a "
     "library with tall wooden shelves, warm afternoon lighting."),
]

NEGATIVE = "lowres, blurry, deformed, jpeg artifacts, worst quality, watermark"
WIDTH = HEIGHT = 1024
STEPS = 25
GUIDANCE = 4.0
BASE_SEED = 42


def gen_batch(pipe, label):
    """Run all prompts through `pipe`, save each as <slug>_<label>.png. Returns list of file paths."""
    print(f"\n=== generating batch: {label} ===")
    out = []
    for i, (slug, prompt) in enumerate(PROMPTS):
        t0 = time.time()
        gen = torch.Generator(device="cuda").manual_seed(BASE_SEED + i)
        try:
            res = pipe(
                prompt=prompt,
                negative_prompt=NEGATIVE,
                num_inference_steps=STEPS,
                guidance_scale=GUIDANCE,
                width=WIDTH, height=HEIGHT,
                generator=gen,
            )
            img = res.images[0]
        except TypeError:
            # Some pipelines reject negative_prompt — retry without it
            gen = torch.Generator(device="cuda").manual_seed(BASE_SEED + i)
            res = pipe(
                prompt=prompt,
                num_inference_steps=STEPS,
                guidance_scale=GUIDANCE,
                width=WIDTH, height=HEIGHT,
                generator=gen,
            )
            img = res.images[0]
        path = OUT_DIR / f"{slug}_{label}.png"
        img.save(path)
        print(f"  {slug}: {path.name} ({time.time()-t0:.1f}s)")
        out.append(path.name)
    return out


def write_index(without_files, with_files):
    """Write a small HTML grid for side-by-side viewing in a browser."""
    rows = []
    for (slug, prompt), wo, wi in zip(PROMPTS, without_files, with_files):
        rows.append(f"""
        <tr>
          <td style="vertical-align:top;padding:8px;width:30%">
            <strong>{slug}</strong><br>
            <small style="color:#666">{prompt}</small>
          </td>
          <td><img src="{wo}" style="width:100%;max-width:380px;border-radius:8px;border:1px solid #ccc"></td>
          <td><img src="{wi}" style="width:100%;max-width:380px;border-radius:8px;border:2px solid #7c4dff"></td>
        </tr>""")
    html = f"""<!doctype html><html><head><meta charset="utf-8">
<title>flammen-aesthetic-v1 A/B</title>
<style>
  body {{ font-family: -apple-system, sans-serif; max-width:1400px; margin:24px auto; padding:0 12px; }}
  th {{ text-align:left; padding:8px; background:#f6f0ff; }}
  td {{ padding:8px; }}
  table {{ border-collapse:collapse; width:100%; }}
  tr {{ border-bottom: 1px solid #eee; }}
</style></head>
<body>
<h1>flammen-aesthetic-v1 — A/B evaluation</h1>
<p>Same prompt, same seed (42 + idx).
Left = base Qwen-Image. Right (purple border) = with flammen-aesthetic-v1 LoRA.</p>
<p><small>Model: {BASE}, LoRA: {LORA_DIR.name}, {STEPS} steps, guidance {GUIDANCE}, {WIDTH}×{HEIGHT}.</small></p>
<table>
  <tr><th>prompt</th><th>without LoRA (baseline)</th><th>with LoRA (purple border)</th></tr>
  {''.join(rows)}
</table>
</body></html>
"""
    (OUT_DIR / "index.html").write_text(html)
    print(f"\nwrote {OUT_DIR / 'index.html'}")


def main():
    print(f"loading {BASE} …")
    pipe = QwenImagePipeline.from_pretrained(BASE, torch_dtype=torch.bfloat16)
    # Qwen-Image total = ~54 GB (38 GB transformer + 14 GB text encoder + 2 GB VAE).
    # Doesn't fit on a 48 GB card with .to("cuda") eager-load. enable_model_cpu_offload
    # parks each component on CPU and only moves the active one to GPU per pipeline
    # stage. Adds ~3–5s of transfer per generation but stable.
    pipe.enable_model_cpu_offload()

    without_files = gen_batch(pipe, "without")

    print(f"\nloading LoRA from {LORA_DIR}")
    pipe.load_lora_weights(str(LORA_DIR), weight_name="pytorch_lora_weights.safetensors")
    with_files = gen_batch(pipe, "with")

    write_index(without_files, with_files)
    print(f"\nA/B done. Open {OUT_DIR}/index.html in a browser.")


if __name__ == "__main__":
    main()
