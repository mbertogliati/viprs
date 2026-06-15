#ifndef VIPRS_H
#define VIPRS_H

#include <stddef.h>
#include <stdint.h>

typedef struct ViprsImage ViprsImage;

ViprsImage* viprs_image_new(
    const uint8_t* data,
    size_t len,
    uint32_t width,
    uint32_t height,
    uint8_t bands
);
void viprs_image_free(ViprsImage* img);
uint32_t viprs_image_width(const ViprsImage* img);
uint32_t viprs_image_height(const ViprsImage* img);
uint8_t viprs_image_bands(const ViprsImage* img);
const uint8_t* viprs_image_data(const ViprsImage* img);

#endif
