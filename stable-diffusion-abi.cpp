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

sd_image_t generate(
        sd_ctx_t *sd_ctx,
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
) {
    sd_image_t *results;

    /*
    sd_ctx_t *sd_ctx = new_sd_ctx(
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
*/

    if (sd_ctx == NULL) {
        LOG_DEBUG("sd_ctx is NULL");
        return results[0];
    }

    results = txt2img(
            sd_ctx,
            "1girl",
            "extra limbs",
            2,
            2,
            768,
            1024,
            static_cast<sample_method_t>(0),
            4,
            4242,
            1,
            nullptr,
            0,
            0,
            false,
            ""
    );

    if (results == NULL) {
        LOG_DEBUG("results is NULL");
        return results[0];
    }

    sd_image_t current_image = results[0];

    LOG_DEBUG("result is %dx%d", current_image.width, current_image.height);


//    if (withUpscale) {
//        upscaler_ctx_t *upscaler_ctx = new_upscaler_ctx(
//                "/media/ed/files/sd/models/ESRGAN/RealESRGAN_x4plus_anime_6B.pth",
//                -1,
//                static_cast<sd_type_t>(1)
//        );
//
//        current_image = upscale(upscaler_ctx, current_image, 2);
//    }

    return current_image;
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