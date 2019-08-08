#include <iostream>
#include <string>
#include <vector>

#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/core/utility.hpp>

#include "app.hpp"
#include "teximage.hpp"
#include "superpixel.hpp"
#include "misc_ocv.hpp"

const unsigned int WIDTH = 432;
const unsigned int HEIGHT = 240;

int main(int, char**) {
    App app = App::Initialize();
    if (!app.ok) return 1;
    GLFWwindow* window = app.window;
    ImGuiIO& io = ImGui::GetIO(); (void)io;
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;  // Enable Keyboard Controls
    ImVec4 clear_color = ImVec4(0.45f, 0.55f, 0.60f, 1.00f);

    Camera cam(0, WIDTH, HEIGHT);
    cam.open();

    cv::Mat frame;
    cv::Mat superpixel_contour;
    cam.capture >> frame;
    cv::Size frame_size = frame.size();
    int width=frame_size.width, height=frame_size.height, channels=3;
    TexImage imSuperpixels(width, height, channels);
#ifdef HAS_LIBGSLIC
    GSLIC _superpixel({
        .img_size = { width, height },
        .no_segs = 64,
        .spixel_size = 32,
        .no_iters = 5,
        .coh_weight = 0.6f,
        .do_enforce_connectivity = true,
        .color_space = gSLICr::XYZ, // gSLICr::CIELAB | gSLICr::RGB
        .seg_method = gSLICr::GIVEN_SIZE // gSLICr::GIVEN_NUM
    });
#endif
    // Main loop
    while (!glfwWindowShouldClose(window)){
        glfwPollEvents();
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();
        {
            ImGui::Begin("Superpixel Analyzer");
            cam.capture >> frame;
#ifdef HAS_LIBGSLIC
            ISuperpixel* superpixel = _superpixel.Compute(frame);
            superpixel->GetContour(superpixel_contour);
            cv::cvtColor(frame, frame, cv::COLOR_BGR2RGB);
#endif
            frame.setTo(cv::Scalar(200, 5, 240), superpixel_contour);
            imSuperpixels.Load(frame.data);
            ImGui::Image(imSuperpixels.id(), imSuperpixels.size(), ImVec2(0,0), ImVec2(1,1), ImVec4(1.0f,1.0f,1.0f,1.0f), ImVec4(1.0f,1.0f,1.0f,0.5f));
            ImGui::End();
        }
        app.Render(clear_color);
    }
    return 0;
}
