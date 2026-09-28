"""Compare atelier's Qwen-Image 2.1 DiT against stable-diffusion.cpp's QwenImage21Runner.

    python scripts/sdcpp_parity/dit_parity.py --harness ./harness [--write-golden]

Runs every case in tests/qwen_image_2_1_cases.py through both implementations.
With --write-golden, stores sd.cpp's outputs in tests/data/qwen_image_2_1_sdcpp.json,
which tests/test_qwen_image_2_1.py checks in CI (no sd.cpp needed there).
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile

import numpy as np
import torch
from safetensors.torch import save_file

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "tests"))
from qwen_image_2_1_cases import CASES, CHANNELS, CONTEXT_DIM, build_case  # noqa: E402

GOLDEN = os.path.join(ROOT, "tests", "data", "qwen_image_2_1_sdcpp.json")


def run_sdcpp(harness, workdir, model, x, t, context, token_types, refs):
    save_file({f"model.diffusion_model.{k}": v.contiguous() for k, v in model.state_dict().items()},
              f"{workdir}/dit.safetensors")
    x.numpy().tofile(f"{workdir}/x.bin")
    context.numpy().tofile(f"{workdir}/ctx.bin")
    token_types.numpy().astype(np.int32).tofile(f"{workdir}/slots.bin")
    ref_args = []
    for i, r in enumerate(refs):
        r.numpy().tofile(f"{workdir}/ref{i}.bin")
        ref_args += [f"{workdir}/ref{i}.bin", str(r.shape[3]), str(r.shape[2])]
    _, _, H, W = x.shape
    cmd = [harness, "dit", f"{workdir}/dit.safetensors", f"{workdir}/x.bin", str(W), str(H), str(CHANNELS),
           str(float(t)), f"{workdir}/ctx.bin", str(CONTEXT_DIM), str(context.shape[1]), f"{workdir}/slots.bin",
           f"{workdir}/out.bin", *ref_args]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        sys.exit(f"harness failed ({r.returncode}):\n{r.stderr[-3000:]}")
    return torch.from_numpy(np.fromfile(f"{workdir}/out.bin", dtype=np.float32)).reshape(1, CHANNELS, H, W)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--harness", default="./harness")
    ap.add_argument("--write-golden", action="store_true")
    args = ap.parse_args()

    golden, worst = {}, 0.0
    with tempfile.TemporaryDirectory() as workdir:
        for name in CASES:
            model, x, t, context, token_types, refs = build_case(name)
            ref = run_sdcpp(args.harness, workdir, model, x, t, context, token_types, refs)
            with torch.no_grad():
                ours = model(x, t, context, token_types, refs)
            rel = ((ours - ref).abs().max() / ref.abs().max()).item()
            worst = max(worst, rel)
            print(f"{name:16s} rel max diff {rel:.2e}")
            golden[name] = ref.flatten().tolist()
    if args.write_golden:
        os.makedirs(os.path.dirname(GOLDEN), exist_ok=True)
        with open(GOLDEN, "w") as f:
            json.dump({k: [round(v, 6) for v in vals] for k, vals in golden.items()}, f)
        print("wrote", GOLDEN)
    sys.exit(0 if worst < 1e-3 else 1)


if __name__ == "__main__":
    main()
