#ifndef STABLE_DIFFUSION_ABI_H
#define STABLE_DIFFUSION_ABI_H

#include "stable-diffusion.h"
#include "ggml_extend.hpp"
#include "utility"

#ifdef STABLE_DIFFUSION_SHARED
#if defined(_WIN32) && !defined(__MINGW32__)
#ifdef STABLE_DIFFUSION_BUILD
#define STABLE_DIFFUSION_API __declspec(dllexport)
#else
#define STABLE_DIFFUSION_API __declspec(dllimport)
#endif
#else
#define STABLE_DIFFUSION_API __attribute__((visibility("default")))
#endif
#else
#define STABLE_DIFFUSION_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

STABLE_DIFFUSION_API sd_image_t *upscale_go(
        upscaler_ctx_t *ctx,
        uint32_t upscale_factor,
        uint32_t width,
        uint32_t height,
        uint32_t channel,
        uint8_t *data
);

typedef struct {
    const char *model_path;
    const char *vae_path;
    const char *taesd_path;
    const char *control_net_path;
    const char *lora_model_dir;
    const char *embed_dir;
    const char *id_embed_dir;
    bool vae_decode_only;
    bool vae_tiling;
    bool free_params_immediately;
    int n_threads;
    enum sd_type_t wType;
    enum rng_type_t rng_type;
    enum schedule_t schedule;
    bool keep_clip_on_cpu;
    bool keep_control_net_cpu;
    bool keep_vae_on_cpu;
} new_sd_ctx_go_params;

STABLE_DIFFUSION_API sd_ctx_t *new_sd_ctx_go(new_sd_ctx_go_params *context_params);

STABLE_DIFFUSION_API struct ggml_context *go_ggml_init(size_t mSize);

STABLE_DIFFUSION_API void go_ggml_tensor_set_f32(struct ggml_tensor *tensor, float value, int l, int k, int j, int i);
STABLE_DIFFUSION_API void go_ggml_tensor_set_f32_randn(struct ggml_tensor *tensor, uint64_t seed);

STABLE_DIFFUSION_API float go_ggml_tensor_get_f32(const ggml_tensor *tensor, int l, int k, int j, int i);

STABLE_DIFFUSION_API void go_ggml_tensor_scale(struct ggml_tensor *src, float scale);
STABLE_DIFFUSION_API void go_ggml_tensor_scale_output(struct ggml_tensor *src);
STABLE_DIFFUSION_API void go_ggml_tensor_clamp(struct ggml_tensor *src, float min, float max);
STABLE_DIFFUSION_API struct ggml_tensor *
go_vector_to_ggml_tensor_i32(struct ggml_context *ctx, const std::vector<int> &vec);

#ifdef __cplusplus
}
#endif

#endif //STABLE_DIFFUSION_ABI_H
