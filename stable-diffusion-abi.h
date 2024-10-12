#ifndef STABLE_DIFFUSION_ABI_H
#define STABLE_DIFFUSION_ABI_H

#include "ggml_extend.hpp"
#include "stable-diffusion.h"
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

typedef struct {
  const char *model_path;
  const char *vae_path;
  const char *taesd_path;
  const char *control_net_path;
  const char *lora_model_dir;
  const char *embed_dir;
  const char *id_embed_dir;

  bool vae_decode_only;
  bool free_params_immediately;
  bool keep_clip_on_cpu;
  bool keep_control_net_cpu;
  bool keep_vae_on_cpu;
  bool vae_tiling;
  bool show_debug;

  uint8_t n_threads;
  uint8_t wType;
  uint8_t rng_type;
  uint8_t schedule;

} new_sd_ctx_go_params;

STABLE_DIFFUSION_API sd_ctx_t *
new_sd_ctx_go(new_sd_ctx_go_params *context_params);

STABLE_DIFFUSION_API sd_image_t *upscale_go(upscaler_ctx_t *ctx,
                                            uint32_t upscale_factor,
                                            uint32_t width, uint32_t height,
                                            uint32_t channel, uint8_t *data);

#ifdef __cplusplus
}
#endif

#endif // STABLE_DIFFUSION_ABI_H
