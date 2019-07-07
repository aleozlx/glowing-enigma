#ifndef __APP_HPP__
#define __APP_HPP__
#include <iostream>
#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include <GL/glew.h>
#include <GLFW/glfw3.h>

/// ImGui App boiler plates
struct App {
    int ok;
    GLFWwindow* window;
    const char* glsl_version;

    inline static App Ok(GLFWwindow* w, const char* glsl_version) {
        return {.ok=1, .window=w, .glsl_version=glsl_version};
    }

    inline static App Err() {
        return {.ok=0, .window=nullptr, .glsl_version=nullptr};
    }

    static App Initialize();
    void Render(ImVec4 &clear_color);
    void Shutdown();
};

#endif
