#include "stable-diffusion-abi.h"
#include "stable-diffusion.h"
#include "util.h"
#include <string>

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
            "",
            "",
            "",
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

struct ggml_context *go_ggml_init(size_t mSize) {
    struct ggml_init_params params;
    params.mem_size = mSize;
    params.mem_size += 2 * ggml_tensor_overhead();
    params.mem_buffer = NULL;
    params.no_alloc = false;

    return ggml_init(params);
}
