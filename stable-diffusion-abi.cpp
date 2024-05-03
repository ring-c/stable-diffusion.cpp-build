#include "stable-diffusion-abi.h"
#include "stable-diffusion.h"
#include "util.h"
#include <string>

uint8_t *get_image_data(const sd_image_t *images, int index) {
    return images[index].data;
}

uint32_t get_image_width(const sd_image_t *images, int index) {
    return images[index].width;
}

uint32_t get_image_height(const sd_image_t *images, int index) {
    return images[index].height;
}

uint32_t get_image_channel(const sd_image_t *images, int index) {
    return images[index].channel;
}

void sd_images_free(const sd_image_t *images) {
    if (images != nullptr) {
        delete[]images;
    }
    images = nullptr;
}

void sd_image_free(sd_image_t *image) {
    if (image != nullptr) {
        delete image;
    }
    image = nullptr;
}

sd_image_t *upscale_go(
        upscaler_ctx_t *ctx,
        uint32_t upscale_factor,
        uint32_t width,
        uint32_t height,
        uint32_t channel,
        uint8_t *data
) {
    sd_image_t *output_image = new(sd_image_t);

    if (ctx == NULL) {
        LOG_DEBUG("ctx is NULL");
        return output_image;
    }

    sd_image_t input_image = sd_image_t{
            .width=width,
            .height=height,
            .channel=channel,
            .data=data,
    };
    
    output_image[0] = upscale(ctx, input_image, upscale_factor);
    return output_image;
}

sd_ctx_t *new_sd_ctx_go(new_sd_ctx_go_params *context_params) {
    return new_sd_ctx(
            context_params->model_path,
            context_params->vae_path,
            context_params->taesd_path,
            context_params->control_net_path,
            context_params->lora_model_dir,
            context_params->embed_dir,
            context_params->id_embed_dir,
            context_params->vae_decode_only,
            context_params->vae_tiling,
            context_params->free_params_immediately,
            context_params->n_threads,
            context_params->wType,
            context_params->rng_type,
            context_params->schedule,
            context_params->keep_clip_on_cpu,
            context_params->keep_control_net_cpu,
            context_params->keep_vae_on_cpu
    );
}

new_sd_ctx_go_params *new_sd_ctx_params(
        const char *model_path,
        const char *lora_model_dir,
        const char *vae_path,
        int16_t n_threads,
        enum sd_type_t wType,
        enum rng_type_t rng_type,
        enum schedule_t schedule
) {
    auto *params = new(new_sd_ctx_go_params);
    params->model_path = model_path;
    params->lora_model_dir = lora_model_dir;
    params->vae_path = vae_path;
    params->n_threads = n_threads;
    params->wType = wType;
    params->rng_type = rng_type;
    params->schedule = schedule;

    // Defaults
    params->taesd_path = "";
    params->control_net_path = "";
    params->embed_dir = "";
    params->id_embed_dir = "";
    params->vae_decode_only = false;
    params->vae_tiling = false;
    params->free_params_immediately = false;
    params->keep_clip_on_cpu = false;
    params->keep_control_net_cpu = false;
    params->keep_vae_on_cpu = false;

    return params;
}

void new_sd_ctx_params_set(
        new_sd_ctx_go_params *params,
        const char *taesd_path,
        const char *control_net_path,
        const char *embed_dir,
        const char *id_embed_dir,
        bool vae_decode_only,
        bool vae_tiling,
        bool free_params_immediately,
        bool keep_clip_on_cpu,
        bool keep_control_net_cpu,
        bool keep_vae_on_cpu
) {
    params->taesd_path = taesd_path;
    params->control_net_path = control_net_path;
    params->embed_dir = embed_dir;
    params->id_embed_dir = id_embed_dir;
    params->vae_decode_only = vae_decode_only;
    params->vae_tiling = vae_tiling;
    params->free_params_immediately = free_params_immediately;
    params->keep_clip_on_cpu = keep_clip_on_cpu;
    params->keep_control_net_cpu = keep_control_net_cpu;
    params->keep_vae_on_cpu = keep_vae_on_cpu;
}