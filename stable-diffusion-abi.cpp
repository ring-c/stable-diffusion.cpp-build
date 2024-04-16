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

// sd_image_t new_image() {
//
// }
//
// void set_image_data(sd_image_t image, uint8_t* data) {
//     image.data = data;
// }
//
// void set_image_width(sd_image_t image, uint32_t width) {
//     image.width = width;
// }
//
// void set_image_height(sd_image_t image, uint32_t height) {
//     imageheight = height;
// }
//
// void set_image_channel(sd_image_t image, uint32_t channel) {
//     image->channel = channel;
// }

sd_image_t *upscale_go(
        upscaler_ctx_t *ctx,
        uint32_t upscale_factor,
        uint32_t width,
        uint32_t height,
        uint32_t channel,
        uint8_t *data
) {
    sd_image_t *output_image = new(sd_image_t);

    sd_image_t input_image = sd_image_t{
            .width=width,
            .height=height,
            .channel=channel,
            .data=data,
    };

    if (ctx == NULL) {
        LOG_DEBUG("ctx is NULL");
        return output_image;
    }

    output_image[0] = upscale(ctx, input_image, upscale_factor);
    return output_image;
}

sd_ctx_t *new_sd_ctx_go() {
    return new_sd_ctx(
            "/media/ed/files/sd/models/Stable-diffusion/dreamshaperXL_v21TurboDPMSDE.safetensors",
            "",
            "",
            "",
            "",
            "",
            "",
            false,
            false,
            true,
            -1,
            static_cast<sd_type_t>(1),
            static_cast<rng_type_t>(1),
            static_cast<schedule_t>(2),
            false,
            false,
            false
    );
}

upscaler_ctx_t *new_upscaler_ctx_go() {
    return new_upscaler_ctx(
            "/media/ed/files/sd/models/ESRGAN/RealESRGAN_x4plus_anime_6B.pth",
            -1,
            static_cast<sd_type_t>(1)
    );
}