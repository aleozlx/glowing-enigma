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
    
    TexImage(unsigned int width, unsigned int height, unsigned int channels);
    inline ImTextureID id() {
        return reinterpret_cast<void*>(texid);
    }
    void Load(const unsigned char *data);
    inline ImVec2 uv(float px=0.0f, float py=0.0f) {
        return ImVec2(px / f32width, py / f32height);
    }
    inline ImVec2 size() {
        return ImVec2(f32width, f32height);
    }
};

#endif
