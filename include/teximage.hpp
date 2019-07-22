#ifndef __TEXIMAGE_HPP__
#define __TEXIMAGE_HPP__
#include "imgui.h"
#include <GL/glew.h>
#include <SOIL/SOIL.h>

/// Management of image data with OpenGL texture
class TexImage {
    private:
    GLuint texid;

    public:
    unsigned int width, height, channels;
    float f32width, f32height;

    /// Create a TexImage with everything uninitialized
    TexImage() {}
    
    TexImage(unsigned int width, unsigned int height, unsigned int channels) {
        this->texid = 0; // The value zero is reserved to represent the default texture for each texture target - Khronos
        this->width = width;
        this->f32width = (float) width;
        this->height = height;
        this->f32height = (float) height;
        this->channels = channels;
    }

    inline ImTextureID id() {
        return reinterpret_cast<void*>(texid);
    }

    void Load(const unsigned char *data) {
        this->texid = SOIL_create_OGL_texture(
            data, width, height, channels,
            this->texid, SOIL_FLAG_MIPMAPS | SOIL_FLAG_NTSC_SAFE_RGB | SOIL_FLAG_COMPRESS_TO_DXT
        );
    }

    inline ImVec2 uv(float px=0.0f, float py=0.0f) {
        return ImVec2(px / f32width, py / f32height);
    }

    inline ImVec2 size() {
        return ImVec2(f32width, f32height);
    }
};

#endif
