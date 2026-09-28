// Parity harness: runs stable-diffusion.cpp's own Qwen-Image 2.1 DiT / VAE
// runners on small weights so Atelier's PyTorch port can be compared against
// them. Links sd.cpp internals; see README.md for build steps.
//
//   harness names <tensor name>...   (prints sd.cpp's converted name, e.g. for LoRA keys)
//   harness dit <weights.safetensors> <x.bin> W H C <t> <ctx.bin> D L <slots.bin> <out.bin> [ref.bin W H]...
//   harness vae <weights.safetensors> <x.bin> W H C <out.bin> [decode]
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <type_traits>
#include <vector>

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "model/diffusion/qwen_image_2_1.hpp"
#include "model/vae/wan_vae.hpp"
#include "model_loader.h"
#include "name_conversion.h"
#include "stable-diffusion.h"

template <typename T>
static std::vector<T> read_bin(const std::string& path, size_t n) {
    std::vector<T> v(n);
    std::ifstream f(path, std::ios::binary);
    f.read(reinterpret_cast<char*>(v.data()), n * sizeof(T));
    if (!f) {
        fprintf(stderr, "short read: %s\n", path.c_str());
        exit(2);
    }
    return v;
}

static void write_bin(const std::string& path, const sd::Tensor<float>& t) {
    std::ofstream f(path, std::ios::binary);
    f.write(reinterpret_cast<const char*>(t.data()), t.numel() * sizeof(float));
    fprintf(stderr, "wrote %s shape [", path.c_str());
    for (auto s : t.shape()) fprintf(stderr, "%lld ", (long long)s);
    fprintf(stderr, "]\n");
}

struct Dit : public Qwen::QwenImage21Runner {
    using Qwen::QwenImage21Runner::QwenImage21Runner;
    void alloc() { ggml_backend_alloc_ctx_tensors(params_ctx, runtime_backend); }
    // Detach the (already allocated + loaded) params from the runner so the
    // segment weight pipeline, which needs a residency manager, ignores them.
    void unmanage() { params_ctx = nullptr; }
};

struct Vae : public WAN::WanVAERunner {
    using WAN::WanVAERunner::WanVAERunner;
    void alloc() { ggml_backend_alloc_ctx_tensors(params_ctx, runtime_backend); }
    // Detach the (already allocated + loaded) params from the runner so the
    // segment weight pipeline, which needs a residency manager, ignores them.
    void unmanage() { params_ctx = nullptr; }
    sd::Tensor<float> run(const sd::Tensor<float>& x, bool decode) { return _compute(4, x, decode); }
};

template <typename R>
static void load(R& runner, ModelLoader& loader, const std::string& prefix = "") {
    runner.alloc();
    std::map<std::string, ggml_tensor*> tensors;
    if constexpr (std::is_same_v<R, Dit>) {
        runner.get_param_tensors(tensors, prefix);
    } else {
        runner.get_param_tensors(tensors);
    }
    if (!loader.load_tensors(tensors)) {
        fprintf(stderr, "load_tensors failed\n");
        exit(3);
    }
    // No DeviceResidencyManager here: params already live in a CPU buffer, so
    // hide them from the segment weight pipeline (it would try to page them).
    runner.unmanage();
}

static void log_cb(enum sd_log_level_t level, const char* text, void*) { fputs(text, stderr); }

int main(int argc, char** argv) {
    if (argc < 3) return 1;
    sd_set_log_callback(log_cb, nullptr);
    std::string mode = argv[1];
    if (mode == "names") {
        for (int i = 2; i < argc; ++i) printf("%s\n", convert_tensor_name(argv[i], VERSION_QWEN_IMAGE_2_1).c_str());
        return 0;
    }
    ModelLoader loader;
    if (!loader.init_from_file(argv[2])) {
        fprintf(stderr, "init_from_file failed\n");
        return 1;
    }
    auto& storage          = loader.get_tensor_storage_map();
    ggml_backend_t backend = ggml_backend_cpu_init();

    if (mode == "dit") {
        int64_t W = atoll(argv[4]), H = atoll(argv[5]), C = atoll(argv[6]);
        float t   = (float)atof(argv[7]);
        int64_t D = atoll(argv[9]), L = atoll(argv[10]);
        sd::Tensor<float> x({W, H, C, 1}, read_bin<float>(argv[3], W * H * C));
        sd::Tensor<float> ts({1}, std::vector<float>{t});
        sd::Tensor<float> ctx({D, L, 1}, read_bin<float>(argv[8], D * L));
        sd::Tensor<int32_t> slots({L}, read_bin<int32_t>(argv[11], L));
        std::vector<sd::Tensor<float>> refs;
        for (int i = 13; i + 2 < argc; i += 3) {
            int64_t rw = atoll(argv[i + 1]), rh = atoll(argv[i + 2]);
            refs.emplace_back(std::vector<int64_t>{rw, rh, C, 1}, read_bin<float>(argv[i], rw * rh * C));
        }
        Dit runner(backend, storage, "model.diffusion_model");
        load(runner, loader, "model.diffusion_model");
        QwenImage21DiffusionExtra extra{&slots};
        DiffusionParams p;
        p.x                            = &x;
        p.timesteps                    = &ts;
        p.context                      = &ctx;
        p.ref_latents                  = &refs;
        p.ref_image_params.pass_to_dit = true;
        p.extra                        = extra;
        auto out                       = runner.compute(4, p);
        if (out.empty()) return 4;
        write_bin(argv[12], out);
    } else if (mode == "vae") {
        int64_t W = atoll(argv[4]), H = atoll(argv[5]), C = atoll(argv[6]);
        bool decode = argc > 8 && std::string(argv[8]) == "decode";
        sd::Tensor<float> x({W, H, C, 1}, read_bin<float>(argv[3], W * H * C));
        Vae runner(backend, storage, "first_stage_model", false, VERSION_QWEN_IMAGE_2_1);
        load(runner, loader);
        auto out = runner.run(x, decode);
        if (out.empty()) return 4;
        write_bin(argv[7], out);
    }
    return 0;
}
