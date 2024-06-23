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

void go_ggml_tensor_set_f32(struct ggml_tensor *tensor, float value, int l, int k = 0, int j = 0, int i = 0) {
    *(float *) ((char *) (tensor->data) + i * tensor->nb[3] + j * tensor->nb[2] + k * tensor->nb[1] +
                l * tensor->nb[0]) = value;
}

void go_ggml_tensor_set_f32_randn(struct ggml_tensor *tensor, uint64_t seed) {
    auto n = (uint32_t) ggml_nelements(tensor);
    std::shared_ptr<RNG> rng = std::make_shared<STDDefaultRNG>();
    rng->manual_seed(seed);
    std::vector<float> random_numbers = rng->randn(n);
    for (uint32_t i = 0; i < n; i++) {
        ggml_set_f32_1d(tensor, i, random_numbers[i]);
    }
}

float go_ggml_tensor_get_f32(const ggml_tensor *tensor, int l, int k, int j, int i) {
    if (tensor->buffer != NULL) {
        float value;
        ggml_backend_tensor_get(tensor, &value,
                                i * tensor->nb[3] + j * tensor->nb[2] + k * tensor->nb[1] + l * tensor->nb[0],
                                sizeof(float));
        return value;
    }
    GGML_ASSERT(tensor->nb[0] == sizeof(float));
    return *(float *) ((char *) (tensor->data) + i * tensor->nb[3] + j * tensor->nb[2] + k * tensor->nb[1] +
                       l * tensor->nb[0]);
}

void go_ggml_tensor_scale(struct ggml_tensor *src, float scale) {
    int64_t nElements = ggml_nelements(src);
    float *data = (float *) src->data;
    for (int i = 0; i < nElements; i++) {
        data[i] = data[i] * scale;
    }
}

void go_ggml_tensor_scale_output(struct ggml_tensor *src) {
    int64_t nelements = ggml_nelements(src);
    float *data = (float *) src->data;
    for (int i = 0; i < nelements; i++) {
        float val = data[i];
        data[i] = (val + 1.0f) * 0.5f;
    }
}

void go_ggml_tensor_clamp(struct ggml_tensor *src, float min, float max) {
    int64_t nElements = ggml_nelements(src);
    float *data = (float *) src->data;
    for (int i = 0; i < nElements; i++) {
        float val = data[i];
        data[i] = val < min ? min : (val > max ? max : val);
    }
}