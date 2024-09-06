#include "stable-diffusion-abi.h"
//#include "stable-diffusion.h"
//#include "util.h"
#include <string>
#include "examples/cli/main.cpp"

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

//    LOG_DEBUG("\n\n\n\n\n");
//    LOG_DEBUG("TEST context_params->model_path %s", context_params->model_path);
//    LOG_DEBUG("TEST context_params->vae_path %s", context_params->vae_path);
//    LOG_DEBUG("TEST context_params->taesd_path %s", context_params->taesd_path);
//    LOG_DEBUG("TEST context_params->control_net_path %s", context_params->control_net_path);
//    LOG_DEBUG("TEST context_params->lora_model_dir %s", context_params->lora_model_dir);
//    LOG_DEBUG("TEST context_params->embed_dir %s", context_params->embed_dir);
//    LOG_DEBUG("TEST context_params->id_embed_dir %s", context_params->id_embed_dir);
//    LOG_DEBUG("TEST context_params->vae_decode_only %d", context_params->vae_decode_only);
//    LOG_DEBUG("TEST context_params->vae_tiling %d", context_params->vae_tiling);
//    LOG_DEBUG("TEST context_params->free_params_immediately %d", context_params->free_params_immediately);
//    LOG_DEBUG("TEST context_params->n_threads %d", context_params->n_threads);
//    LOG_DEBUG("TEST context_params->wType %d", context_params->wType);
//    LOG_DEBUG("TEST context_params->rng_type %d", context_params->rng_type);
//    LOG_DEBUG("TEST context_params->schedule %d", context_params->schedule);
//    LOG_DEBUG("TEST context_params->keep_clip_on_cpu %d", context_params->keep_clip_on_cpu);
//    LOG_DEBUG("TEST context_params->keep_control_net_cpu %d", context_params->keep_control_net_cpu);
//    LOG_DEBUG("TEST context_params->keep_vae_on_cpu %d", context_params->keep_vae_on_cpu);
//    LOG_DEBUG("\n\n\n\n\n");

    SDParams params;
    params.verbose = true;
    params.color = true;
    sd_set_log_callback(sd_log_cb, &params);

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
            sd_type_t(context_params->wType),
            rng_type_t(context_params->rng_type),
            schedule_t(context_params->schedule),
            context_params->keep_clip_on_cpu,
            context_params->keep_control_net_cpu,
            context_params->keep_vae_on_cpu
    );
}
