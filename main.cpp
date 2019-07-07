#include <iostream>
#include <string>
#include <vector>

#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/core/utility.hpp>

#include "superpixel_pipeline.hpp"
#include "teximage.hpp"

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

struct RGBHistogram {
    float alpha;
    unsigned int width, height;
    std::vector<cv::Mat> bgr_planes;

    RGBHistogram(cv::InputArray inputRGBImage, unsigned int width, unsigned int height, float alpha=1.0f) {
        this->width = width;
        this->height = height;
        this->alpha = alpha;
        cv::split(inputRGBImage, bgr_planes);
    }

    void Compute(cv::OutputArray output, cv::InputArray mask=cv::noArray()) {
        const int histSize = 256;
        const float range[] = { 0, 256 } ;
        const float* histRange = { range };
        bool uniform = true; bool accumulate = false;
        cv::Mat b_hist, g_hist, r_hist;
        cv::calcHist(&bgr_planes[0], 1, 0, cv::noArray(), b_hist, 1, &histSize, &histRange, uniform, accumulate);
        cv::calcHist(&bgr_planes[1], 1, 0, cv::noArray(), g_hist, 1, &histSize, &histRange, uniform, accumulate);
        cv::calcHist(&bgr_planes[2], 1, 0, cv::noArray(), r_hist, 1, &histSize, &histRange, uniform, accumulate);

        // Draw the global histograms for B, G and R
        int bin_w = cvRound((double) width/histSize);
        cv::Mat histImage(height, width, CV_8UC3, cv::Scalar(0,0,0));

        /// Normalize the result to [ 0, histImage.rows ]
        cv::normalize(b_hist, b_hist, 0, histImage.rows, cv::NORM_MINMAX, -1, cv::noArray());
        cv::normalize(g_hist, g_hist, 0, histImage.rows, cv::NORM_MINMAX, -1, cv::noArray());
        cv::normalize(r_hist, r_hist, 0, histImage.rows, cv::NORM_MINMAX, -1, cv::noArray());

        // Draw for each channel
        // Note: Blue and Red channels are purposefully swapped to avoid color conversion before rendering
        float _alpha = mask.empty()?1.0f:this->alpha;
        for(int i = 1;i < histSize;i++){
            cv::line( histImage, cv::Point( bin_w*(i-1), height - cvRound(b_hist.at<float>(i-1)) ) ,
                            cv::Point( bin_w*(i), height - cvRound(b_hist.at<float>(i)) ),
                            cv::Scalar( 0, 0, 255*_alpha), 2, 8, 0  );
            cv::line( histImage, cv::Point( bin_w*(i-1), height - cvRound(g_hist.at<float>(i-1)) ) ,
                            cv::Point( bin_w*(i), height - cvRound(g_hist.at<float>(i)) ),
                            cv::Scalar( 0, 255*_alpha, 0), 2, 8, 0  );
            cv::line( histImage, cv::Point( bin_w*(i-1), height - cvRound(r_hist.at<float>(i-1)) ) ,
                            cv::Point( bin_w*(i), height - cvRound(r_hist.at<float>(i)) ),
                            cv::Scalar( 255*_alpha, 0, 0), 2, 8, 0  );
        }

        if (!mask.empty()) {
            cv::calcHist(&bgr_planes[0], 1, 0, mask, b_hist, 1, &histSize, &histRange, uniform, accumulate);
            cv::calcHist(&bgr_planes[1], 1, 0, mask, g_hist, 1, &histSize, &histRange, uniform, accumulate);
            cv::calcHist(&bgr_planes[2], 1, 0, mask, r_hist, 1, &histSize, &histRange, uniform, accumulate);
            // cv::normalize(b_hist, b_hist, 0, histImage.rows, cv::NORM_MINMAX, -1, cv::Mat());
            // cv::normalize(g_hist, g_hist, 0, histImage.rows, cv::NORM_MINMAX, -1, cv::Mat());
            // cv::normalize(r_hist, r_hist, 0, histImage.rows, cv::NORM_MINMAX, -1, cv::Mat());
            for(int i = 1;i < histSize;i++){
                cv::line( histImage, cv::Point( bin_w*(i-1), height - cvRound(b_hist.at<float>(i-1)) ) ,
                                cv::Point( bin_w*(i), height - cvRound(b_hist.at<float>(i)) ),
                                cv::Scalar( 0, 0, 255), 2, 8, 0  );
                cv::line( histImage, cv::Point( bin_w*(i-1), height - cvRound(g_hist.at<float>(i-1)) ) ,
                                cv::Point( bin_w*(i), height - cvRound(g_hist.at<float>(i)) ),
                                cv::Scalar( 0, 255, 0), 2, 8, 0  );
                cv::line( histImage, cv::Point( bin_w*(i-1), height - cvRound(r_hist.at<float>(i-1)) ) ,
                                cv::Point( bin_w*(i), height - cvRound(r_hist.at<float>(i)) ),
                                cv::Scalar( 255, 0, 0), 2, 8, 0  );
            }
        }
        output.assign(histImage);
    }
};

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
    TexImage imSuperpixels(width, height, channels);
    TexImage imHistogram(432, 400, 3);
    cv::Mat histogram, histogram_rgb;

#define USE_OCVSLIC
#ifdef HAS_LIBGSLIC
#undef USE_OCVSLIC
#define USE_GSLIC
#ifdef USE_GSLIC
    gSLICr::objects::settings my_settings;
	my_settings.img_size.x = 432;
	my_settings.img_size.y = 240;
	// my_settings.no_segs = 2000;
	my_settings.spixel_size = 32;
	my_settings.coh_weight = 0.6f;
	my_settings.no_iters = 5;
	my_settings.color_space = gSLICr::XYZ; // gSLICr::CIELAB for Lab, or gSLICr::RGB for RGB
	my_settings.seg_method = gSLICr::GIVEN_SIZE; // or gSLICr::GIVEN_NUM for given number
	my_settings.do_enforce_connectivity = true; // whether or not run the enforce connectivity step
    GSLIC _superpixel(my_settings);
#endif
#endif

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
            static bool use_spotlight = false;
            static bool use_magnifier = false;

            ImGui::Begin("Superpixel Analyzer");

            cam.capture >> frame;
            cv::cvtColor(frame, frame_rgb, cv::COLOR_BGR2RGB);
            cv::medianBlur(frame, frame_hsv, 5);
            cv::cvtColor(frame_hsv, frame_hsv, cv::COLOR_BGR2HSV);
#ifdef USE_OCVSLIC
            OpenCVSLIC _superpixel(frame_hsv, superpixel_size, 30.0f, 3, 10.0f);
#endif
#ifdef USE_GSLIC
            _superpixel.with(frame);
#endif
            ISuperpixel* superpixel = _superpixel.Compute();
            std::cout<<"Compute()e"<<std::endl;

            superpixel->GetContour(superpixel_contour);
            std::cout<<"GetContour()e"<<std::endl;
            // superpixel->GetLabels(superpixel_labels);
            // superpixel_id = superpixel_labels.at<unsigned int>(pointer_y, pointer_x);
            // superpixel_selected = superpixel_labels == superpixel_id;
            // cv::meanStdDev(frame_rgb, sel_mean, sel_std, superpixel_selected);
            // if (use_spotlight) spotlight(frame_rgb, superpixel_selected, 0.5);
            // frame_rgb.setTo(cv::Scalar(200, 5, 240), superpixel_contour);
            
            imSuperpixels.Load(frame_rgb.data);
            ImGui::Text("(%d, %d) => (%d,)", imSuperpixels.width, imSuperpixels.height, superpixel->GetNumSuperpixels());
            ImVec2 pos = ImGui::GetCursorScreenPos();
            ImGui::Image(imSuperpixels.id(), imSuperpixels.size(), ImVec2(0,0), ImVec2(1,1), ImVec4(1.0f,1.0f,1.0f,1.0f), ImVec4(1.0f,1.0f,1.0f,0.5f));
            if (ImGui::IsItemHovered()) {
                pointer_x = static_cast<int>(io.MousePos.x - pos.x);
                pointer_y = static_cast<int>(io.MousePos.y - pos.y);
                if (use_magnifier) {
                    ImGui::BeginTooltip();
                    float region_sz = 32.0f;
                    float region_x = io.MousePos.x - pos.x - region_sz * 0.5f; if (region_x < 0.0f) region_x = 0.0f; else if (region_x > imSuperpixels.f32width - region_sz) region_x = imSuperpixels.f32width - region_sz;
                    float region_y = io.MousePos.y - pos.y - region_sz * 0.5f; if (region_y < 0.0f) region_y = 0.0f; else if (region_y > imSuperpixels.f32height - region_sz) region_y = imSuperpixels.f32height - region_sz;
                    float zoom = 4.0f;
                    ImGui::Text("Ptr: (%d,%d) Id: %d", pointer_x, pointer_y, superpixel_id);
                    ImGui::Text("Mean: (%.1f,%.1f,%.1f)", sel_mean[0], sel_mean[1], sel_mean[2]);
                    ImGui::Text("Std: (%.1f,%.1f,%.1f)", sel_std[0], sel_std[1], sel_std[2]);
                    ImGui::Image(
                        imSuperpixels.id(), ImVec2(region_sz * zoom, region_sz * zoom),
                        imSuperpixels.uv(region_x, region_y), imSuperpixels.uv(region_x + region_sz, region_y + region_sz),
                        ImVec4(1.0f, 1.0f, 1.0f, 1.0f), ImVec4(1.0f, 1.0f, 1.0f, 0.5f));
                    ImGui::EndTooltip();
                }
            }
            ImGui::SliderFloat("Size", &superpixel_size, 15.0f, 80.0f);
            ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / ImGui::GetIO().Framerate, ImGui::GetIO().Framerate);
            ImGui::Checkbox("Spotlight", &use_spotlight); ImGui::SameLine(150);
            ImGui::Checkbox("Magnifier", &use_magnifier);

            RGBHistogram hist(frame, imHistogram.width, imHistogram.height, (use_spotlight?0.3f:1.0f));
            hist.Compute(histogram, use_spotlight?superpixel_selected:cv::noArray());
            imHistogram.Load(histogram.data);
            ImGui::Image(imHistogram.id(), imHistogram.size(), ImVec2(0,0), ImVec2(1,1), ImVec4(1.0f,1.0f,1.0f,1.0f), ImVec4(1.0f,1.0f,1.0f,0.5f));
            ImGui::End();
        }
        app.Render(clear_color);
    }
    app.Shutdown();
    return 0;
}
