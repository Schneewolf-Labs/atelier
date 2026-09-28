# stable-diffusion.cpp parity checks

diffusers has no Qwen-Image 2.1, so Atelier's port (`atelier/models/qwen_image_2_1.py`)
is checked against [stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp)'s
implementation directly: `harness.cpp` runs sd.cpp's own `QwenImage21Runner` and
`WanVAERunner` on random / deterministic weights, and the scripts compare outputs.

```bash
# 1. build sd.cpp (CPU is fine)
git -C stable-diffusion.cpp submodule update --init ggml
cmake -S stable-diffusion.cpp -B sdbuild -DCMAKE_BUILD_TYPE=Release -DSD_BUILD_EXAMPLES=OFF
cmake --build sdbuild -j

# 2. build the harness against it
S=stable-diffusion.cpp B=sdbuild
c++ -O2 -std=gnu++17 -DGGML_MAX_NAME=160 -DGGML_USE_CPU \
  -I$S/ggml/src -I$S -I$S/src -I$S/include -I$S/src/core -I$S/thirdparty -I$S/ggml/include \
  scripts/sdcpp_parity/harness.cpp -o harness \
  $B/libstable-diffusion.a $(find $B/thirdparty -name '*.a') \
  $B/ggml/src/libggml.a $B/ggml/src/libggml-cpu.a $B/ggml/src/libggml-base.a -lpthread -fopenmp

# 3. compare
python scripts/sdcpp_parity/dit_parity.py --harness ./harness              # ~3e-4 rel
python scripts/sdcpp_parity/dit_parity.py --harness ./harness --write-golden  # refresh CI golden
python scripts/sdcpp_parity/vae_parity.py --harness ./harness --kernels 2d  # ~5e-3 rel (sd.cpp fp16 convs)
python scripts/sdcpp_parity/vae_parity.py --harness ./harness --kernels 3d
./harness names lora.diffusion_model.transformer_blocks.0.attn.to_q.lora_A.weight   # LoRA key mapping
```

`tests/data/qwen_image_2_1_sdcpp.json` holds sd.cpp's DiT outputs for the cases in
`tests/qwen_image_2_1_cases.py`, so CI re-checks parity without building sd.cpp.
Refresh it whenever sd.cpp's implementation changes.

The harness links sd.cpp internals (runner classes, `ModelLoader`), so it tracks
sd.cpp's source layout; last built and verified against sd.cpp `2bb7294`. It sidesteps
sd.cpp's device-residency manager by loading params into a plain CPU buffer and
detaching them from the runner (`unmanage()`), which is fine for these sizes.
