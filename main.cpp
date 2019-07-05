#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include <iostream>
#include <string>
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <SOIL/SOIL.h>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/core/utility.hpp>
#include "superpixel_pipeline.hpp"

static void glfw_error_callback(int error, const char* description) {
    std::cerr << "Glfw Error " << error << description << std::endl;
}

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

    void Render(ImVec4 &clear_color) {
        ImGui::Render();
        int display_w, display_h;
        glfwMakeContextCurrent(window);
        glfwGetFramebufferSize(window, &display_w, &display_h);
        glViewport(0, 0, display_w, display_h);
        glClearColor(clear_color.x, clear_color.y, clear_color.z, clear_color.w);
        glClear(GL_COLOR_BUFFER_BIT);
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

        glfwMakeContextCurrent(window);
        glfwSwapBuffers(window);
    }

    void Shutdown() {
        ImGui_ImplOpenGL3_Shutdown();
        ImGui_ImplGlfw_Shutdown();
        ImGui::DestroyContext();
        glfwDestroyWindow(this->window);
        glfwTerminate();
    }

    static App Initialize() {
        glfwSetErrorCallback(glfw_error_callback);
        if (!glfwInit()) return App::Err();

        // GL 3.0 + GLSL 130
        const char* glsl_version = "#version 130";
        glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
        glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);
        //glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);  // 3.2+ only
        //glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);            // 3.0+ only

        // Create window with graphics context
        GLFWwindow* window = glfwCreateWindow(1280, 720, "Superpixel Analyzer", NULL, NULL);
        if (window == NULL) return App::Err();
        glfwMakeContextCurrent(window);
        glfwSwapInterval(1); // ? "Enable vsync" - what is this?

        bool err = glewInit() != GLEW_OK;
        if (err) {
            std::cerr << "Failed to initialize OpenGL loader!" << std::endl;
            return App::Err();
        }

        // Setup Dear ImGui context
        IMGUI_CHECKVERSION();
        ImGui::CreateContext();

        // Setup Dear ImGui style
        ImGui::StyleColorsDark();
        //ImGui::StyleColorsClassic();

        // Setup Platform/Renderer bindings
        ImGui_ImplGlfw_InitForOpenGL(window, true);
        ImGui_ImplOpenGL3_Init(glsl_version);

        return App::Ok(window, glsl_version);
    }
};

/// OpenCV Video I/O
struct Camera {
    int capture_id;
    cv::VideoCapture capture;
    unsigned int width, height;

    Camera(unsigned int idx, unsigned int width, unsigned int height): capture_id(idx), capture(idx) {
        this->width = width;
        this->height = height;
    }

    int open() {
        if (!capture.open(capture_id)) {
            std::cerr << "Cannot initialize video:" << capture_id << std::endl;
            return 0;
        }
        else {
            capture.set(cv::CAP_PROP_FRAME_WIDTH, width);
            capture.set(cv::CAP_PROP_FRAME_HEIGHT, height);
            return 1;
        }
    }
};

/// Management of image data in texture
class TexImage {
    private:
    bool _firstCall;
    GLuint texid;

    public:
    unsigned int width, height, channels;
    float f32width, f32height;

    TexImage(unsigned int width, unsigned int height, unsigned int channels) {
        this->_firstCall = true;
        this->texid = 0;
        this->width = width;
        this->f32width = (float) width;
        this->height = height;
        this->f32height = (float) height;
        this->channels = channels;
    }

    void Load(const unsigned char *data) {
        GLuint ret = SOIL_create_OGL_texture(
            data, width, height, channels,
            this->_firstCall?SOIL_CREATE_NEW_ID:texid,
            SOIL_FLAG_MIPMAPS | SOIL_FLAG_NTSC_SAFE_RGB | SOIL_FLAG_COMPRESS_TO_DXT
        );
        if (this->_firstCall) {
            texid = ret;
            this->_firstCall = false;
        }
    }

    inline ImTextureID id() {
        return reinterpret_cast<void*>(texid);
    }

    ImVec2 uv(float px=0.0f, float py=0.0f) {
        return ImVec2(px / f32width, py / f32height);
    }

    ImVec2 size() {
        return ImVec2(f32width, f32height);
    }
};


std::string cv_type2str(int type) {
    /*
    std::string ty = cv_type2str(frame_rgb.type());
    ImGui::Text("type %s", ty.c_str());
    */
    std::string r;
    unsigned char depth = type & CV_MAT_DEPTH_MASK;
    unsigned char chans = 1 + (type >> CV_CN_SHIFT);
    switch (depth) {
        case CV_8U:  r = "8U"; break;
        case CV_8S:  r = "8S"; break;
        case CV_16U: r = "16U"; break;
        case CV_16S: r = "16S"; break;
        case CV_32S: r = "32S"; break;
        case CV_32F: r = "32F"; break;
        case CV_64F: r = "64F"; break;
        default:     r = "User"; break;
    }
    r += "C";
    r += (chans+'0');
    return r;
}

void spotlight(cv::OutputArray _frame, cv::InputArray _sel, float alpha) {
    cv::Mat frame = _frame.getMat(), selection = _sel.getMat();
    CV_Assert(frame.type() == CV_8UC3 && selection.type() == CV_8UC1);
    const int channels = 3;
    for (int i=0; i<frame.rows; ++i) {
        unsigned char* fptr = frame.ptr(i);
        const unsigned char* sptr = selection.ptr(i);
        for (int j=0; j<frame.cols; ++j) {
            if (sptr[j]==0) {
                fptr[j*channels] = static_cast<unsigned char>(fptr[j*channels] * alpha);
                fptr[j*channels+1] = static_cast<unsigned char>(fptr[j*channels+1] * alpha);
                fptr[j*channels+2] = static_cast<unsigned char>(fptr[j*channels+2] * alpha);
            }
        }
    }
}

int main(int, char**) {
    App app = App::Initialize();
    if (!app.ok) return 1;
    GLFWwindow* window = app.window;
    ImGuiIO& io = ImGui::GetIO(); (void)io;
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;  // Enable Keyboard Controls
    ImVec4 clear_color = ImVec4(0.45f, 0.55f, 0.60f, 1.00f);

    Camera cam(0, 432, 240);
    cam.open();

    cv::Mat frame, frame_rgb, frame_hsv;
    cv::Mat superpixel_contour, superpixel_labels, superpixel_selected;
    cv::Scalar sel_mean, sel_std;
    cam.capture >> frame;
    cv::Size frame_size = frame.size();
    int width=frame_size.width, height=frame_size.height, channels=3;
    TexImage my_tex(width, height, channels);

    // Main loop
    while (!glfwWindowShouldClose(window)){
        glfwPollEvents();

        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();
        {
            static float superpixel_size = 32.0f;
            static int pointer_x = 0, pointer_y = 0;
            static unsigned int superpixel_id = 0;
            static bool use_spotlight = true;
            static bool use_magnifier = false;

            ImGui::Begin("Superpixel Analyzer");

            cam.capture >> frame;
            cv::cvtColor(frame, frame_rgb, cv::COLOR_BGR2RGB);
            cv::medianBlur(frame, frame_hsv, 5);
            cv::cvtColor(frame_hsv, frame_hsv, cv::COLOR_BGR2HSV);
            OpenCVSLIC _superpixel(frame_hsv, superpixel_size, 30.0f, 3, 10.0f);
            ISuperpixel* superpixel = _superpixel.Compute();

            superpixel->GetContour(superpixel_contour);
            superpixel->GetLabels(superpixel_labels);
            superpixel_id = superpixel_labels.at<unsigned int>(pointer_y, pointer_x);
            superpixel_selected = superpixel_labels == superpixel_id;
            cv::meanStdDev(frame_rgb, sel_mean, sel_std, superpixel_selected);
            if (use_spotlight) spotlight(frame_rgb, superpixel_selected, 0.5);
            frame_rgb.setTo(cv::Scalar(200, 5, 240), superpixel_contour);
            
            my_tex.Load(frame_rgb.data);
            ImGui::Text("(%d, %d) => (%d,)", my_tex.width, my_tex.height, superpixel->GetNumSuperpixels());
            ImVec2 pos = ImGui::GetCursorScreenPos();
            ImGui::Image(my_tex.id(), my_tex.size(), ImVec2(0,0), ImVec2(1,1), ImVec4(1.0f,1.0f,1.0f,1.0f), ImVec4(1.0f,1.0f,1.0f,0.5f));
            if (ImGui::IsItemHovered()) {
                pointer_x = static_cast<int>(io.MousePos.x - pos.x);
                pointer_y = static_cast<int>(io.MousePos.y - pos.y);
                if (use_magnifier) {
                    ImGui::BeginTooltip();
                    float region_sz = 32.0f;
                    float region_x = io.MousePos.x - pos.x - region_sz * 0.5f; if (region_x < 0.0f) region_x = 0.0f; else if (region_x > my_tex.f32width - region_sz) region_x = my_tex.f32width - region_sz;
                    float region_y = io.MousePos.y - pos.y - region_sz * 0.5f; if (region_y < 0.0f) region_y = 0.0f; else if (region_y > my_tex.f32height - region_sz) region_y = my_tex.f32height - region_sz;
                    float zoom = 4.0f;
                    ImGui::Text("Ptr: (%d,%d) Id: %d", pointer_x, pointer_y, superpixel_id);
                    ImGui::Text("Mean: (%.1f,%.1f,%.1f)", sel_mean[0], sel_mean[1], sel_mean[2]);
                    ImGui::Text("Std: (%.1f,%.1f,%.1f)", sel_std[0], sel_std[1], sel_std[2]);
                    ImGui::Image(
                        my_tex.id(), ImVec2(region_sz * zoom, region_sz * zoom),
                        my_tex.uv(region_x, region_y), my_tex.uv(region_x + region_sz, region_y + region_sz),
                        ImVec4(1.0f, 1.0f, 1.0f, 1.0f), ImVec4(1.0f, 1.0f, 1.0f, 0.5f));
                    ImGui::EndTooltip();
                }
            }
            ImGui::SliderFloat("Size", &superpixel_size, 15.0f, 80.0f);
            ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / ImGui::GetIO().Framerate, ImGui::GetIO().Framerate);
            ImGui::Checkbox("Spotlight", &use_spotlight); ImGui::SameLine(150);
            ImGui::Checkbox("Magnifier", &use_magnifier);
            ImGui::End();
        }
        app.Render(clear_color);
    }
    app.Shutdown();
    return 0;
}
