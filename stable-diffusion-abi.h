#ifndef STABLE_DIFFUSION_ABI_H
#define STABLE_DIFFUSION_ABI_H

#include "stable-diffusion.h"

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

// STABLE_DIFFUSION_API void set_image_data(sd_image_t* image, uint8_t* data);
//
// STABLE_DIFFUSION_API void set_image_width(sd_image_t* image, uint32_t width);
//
// STABLE_DIFFUSION_API void set_image_height(sd_image_t* image, uint32_t height);
//
// STABLE_DIFFUSION_API void set_image_channel(sd_image_t* image, uint32_t channel);

STABLE_DIFFUSION_API uint8_t *get_image_data(const sd_image_t *images, int index);

STABLE_DIFFUSION_API uint32_t get_image_width(const sd_image_t *images, int index);

STABLE_DIFFUSION_API uint32_t get_image_height(const sd_image_t *images, int index);

STABLE_DIFFUSION_API uint32_t get_image_channel(const sd_image_t *images, int index);

STABLE_DIFFUSION_API void sd_images_free(const sd_image_t *images);

STABLE_DIFFUSION_API void sd_image_free(sd_image_t *image);


STABLE_DIFFUSION_API sd_image_t *generate(
        sd_ctx_t *sd_ctx,
        upscaler_ctx_t *upscaler_ctx,
        int clip_skip,
        float cfg_scale,
        int width,
        int height,
        enum sample_method_t sample_method,
        int sample_steps,
        int64_t seed,
        int batch_count,
        bool withUpscale,
        int upscaleScale
);

STABLE_DIFFUSION_API sd_ctx_t *new_sd_ctx_go();
STABLE_DIFFUSION_API upscaler_ctx_t *new_upscaler_ctx_go();

#ifdef __cplusplus
}
#endif

#endif //STABLE_DIFFUSION_ABI_H
