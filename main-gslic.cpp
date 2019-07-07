#include <iostream>
#include <string>
#include <vector>

#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/core/utility.hpp>

#include "app.hpp"
#include "teximage.hpp"
#include "superpixel_pipeline.hpp"
#include "misc_ocv.hpp"

int main(int, char**) {
    App app = App::Initialize();
    if (!app.ok) return 1;
    GLFWwindow* window = app.window;
    ImGuiIO& io = ImGui::GetIO(); (void)io;
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;  // Enable Keyboard Controls
    ImVec4 clear_color = ImVec4(0.45f, 0.55f, 0.60f, 1.00f);

    Camera cam(0, 432, 240);
    cam.open();

    cv::Mat frame;
    cv::Mat superpixel_contour;
    cam.capture >> frame;
    cv::Size frame_size = frame.size();
    int width=frame_size.width, height=frame_size.height, channels=3;
    TexImage imSuperpixels(width, height, channels);
    TexImage imHistogram(432, 400, 3);
    cv::Mat histogram, histogram_rgb;


    gSLICr::objects::settings my_settings;
	my_settings.img_size.x = 432;
	my_settings.img_size.y = 240;
	// my_settings.no_segs = 200;
	my_settings.spixel_size = 32;
	my_settings.coh_weight = 0.6f;
	my_settings.no_iters = 5;
	my_settings.color_space = gSLICr::XYZ; // gSLICr::CIELAB for Lab, or gSLICr::RGB for RGB
	my_settings.seg_method = gSLICr::GIVEN_SIZE; // or gSLICr::GIVEN_NUM for given number
	my_settings.do_enforce_connectivity = true;
    GSLIC _superpixel(my_settings);

    // Main loop
    while (!glfwWindowShouldClose(window)){
        glfwPollEvents();

        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();
        {
            ImGui::Begin("Superpixel Analyzer");

            cam.capture >> frame;

            _superpixel.with(frame);
            ISuperpixel* superpixel = _superpixel.Compute();
            superpixel->GetContour(superpixel_contour);
            
            imSuperpixels.Load(superpixel_contour.data);
            ImGui::Text("(%d, %d) => (%d,)", imSuperpixels.width, imSuperpixels.height, 0);
            ImGui::Image(imSuperpixels.id(), imSuperpixels.size(), ImVec2(0,0), ImVec2(1,1), ImVec4(1.0f,1.0f,1.0f,1.0f), ImVec4(1.0f,1.0f,1.0f,0.5f));

            // ImGui::SliderFloat("Size", &superpixel_size, 15.0f, 80.0f);
            ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / ImGui::GetIO().Framerate, ImGui::GetIO().Framerate);

            ImGui::End();
        }
        app.Render(clear_color);
    }
    app.Shutdown();
    return 0;
}
