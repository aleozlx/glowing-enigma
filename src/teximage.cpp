#include "teximage.hpp"

TexImage::TexImage(unsigned int width, unsigned int height, unsigned int channels) {
    this->texid = 0; // The value zero is reserved to represent the default texture for each texture target - Khronos
    this->width = width;
    this->f32width = (float) width;
    this->height = height;
    this->f32height = (float) height;
    this->channels = channels;
}

void TexImage::Load(const unsigned char *data) {
    this->texid = SOIL_create_OGL_texture(
        data, width, height, channels,
        this->texid, SOIL_FLAG_MIPMAPS | SOIL_FLAG_NTSC_SAFE_RGB | SOIL_FLAG_COMPRESS_TO_DXT
    );
}
