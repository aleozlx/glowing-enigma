#include <iostream>
#include <string>
#include <vector>
#include <tuple>

#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/core/utility.hpp>

#include "app.hpp"
#include "teximage.hpp"
#include "superpixel_pipeline.hpp"
#include "misc_ocv.hpp"

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

// TODO investigate ImGui::PlotHistogram
struct RGBHistogram {
    float alpha;
    unsigned int width, height;
    std::vector<cv::Mat> bgr_planes;
    bool normalize_component;

    RGBHistogram(cv::InputArray inputRGBImage, unsigned int width, unsigned int height, float alpha=1.0f, bool normalize_component=false) {
        this->width = width;
        this->height = height;
        this->alpha = alpha;
        this->normalize_component = normalize_component;
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
        float bin_w = (double)width / histSize;
        cv::Mat histImage(height, width, CV_8UC3, cv::Scalar(0,0,0));

        // Normalize the result
        double mb = cv::norm(b_hist, cv::NORM_INF),
            mg = cv::norm(g_hist, cv::NORM_INF),
            mr = cv::norm(r_hist, cv::NORM_INF);
        double max_count = std::max({mb, mg, mr});
        b_hist = b_hist / max_count * histImage.rows;
        g_hist = g_hist / max_count * histImage.rows;
        r_hist = r_hist / max_count * histImage.rows;
        
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
            if (normalize_component) {
                cv::normalize(b_hist, b_hist, 0, histImage.rows, cv::NORM_MINMAX, -1, cv::noArray());
                cv::normalize(g_hist, g_hist, 0, histImage.rows, cv::NORM_MINMAX, -1, cv::noArray());
                cv::normalize(r_hist, r_hist, 0, histImage.rows, cv::NORM_MINMAX, -1, cv::noArray());
            }
            else {
                b_hist = b_hist / max_count * histImage.rows;
                g_hist = g_hist / max_count * histImage.rows;
                r_hist = r_hist / max_count * histImage.rows;
            }
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

const unsigned int WIDTH = 432;
const unsigned int HEIGHT = 240;

int main(int, char**) {
    App app = App::Initialize();
    if (!app.ok) return 1;
    GLFWwindow* window = app.window;
    ImGuiIO& io = ImGui::GetIO(); (void)io;
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;  // Enable Keyboard Controls
    ImVec4 clear_color = ImVec4(0.45f, 0.55f, 0.60f, 1.00f);

    std::vector<CameraInfo> cameras;
    for (auto const v: cv_misc::camera_enumerate()) {
        char camera_name[20];
        std::snprintf(camera_name, IM_ARRAYSIZE(camera_name), "/dev/video%d", v);
        cameras.push_back({ .name = std::string(camera_name), .id = v });
    }
    Camera cam(0, WIDTH, HEIGHT);
    cam.open();

    cv::Mat frame, frame_rgb;
    cv::Mat superpixel_contour, superpixel_labels, superpixel_selected;
    cv::Scalar sel_mean, sel_std;
    cam.capture >> frame;
    cv::Size frame_size = frame.size();
    int width=frame_size.width, height=frame_size.height, channels=3;
    TexImage imSuperpixels(width, height, channels);
    TexImage imHistogram(width - 20, height, channels);
    cv::Mat histogram, histogram_rgb;
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
#else
    OpenCVSLIC _superpixel(32, 30.0f, 3, 10.0f);
#endif

    // Main loop
    while (!glfwWindowShouldClose(window)){
        glfwPollEvents();

        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();
        {
            // **********************
            // * Pipeline Settings window
            // **********************
            ImGui::Begin("Pipeline Settings");
            if(ImGui::TreeNode("Enabled Features")) {
                #define FEATURE(FEATURE_MACRO) ImGui::Text("-D " #FEATURE_MACRO)
                #define FEATURE_VER(FEATURE_MACRO) ImGui::Text("-D " #FEATURE_MACRO " %s", VERSTR(FEATURE_MACRO))
                #include "features.hpp"
                ImGui::TreePop();
            }

            if(ImGui::TreeNode("Video Feed") && cameras.size()>0) {
                static const CameraInfo *d_camera_current = &cameras[0];
                if (ImGui::BeginCombo("Source", d_camera_current->name.c_str())) {
                    for (auto const &camera_info: cameras) {
                        bool is_selected = (d_camera_current == &camera_info);
                        if (ImGui::Selectable(camera_info.name.c_str(), is_selected))
                            d_camera_current = &camera_info;
                        if (is_selected)
                            ImGui::SetItemDefaultFocus();
                    }
                    ImGui::EndCombo();
                }
                ImGui::TreePop();
            }

#ifdef HAS_LIBGSLIC // with gSLIC, it is not efficient to use different superpixel sizes over time
            if(ImGui::TreeNode("Superpixels")) {
                static float d_superpixel_size = 32.0f;
                ImGui::SliderFloat("Superpixel Size", &d_superpixel_size, 15.0f, 80.0f);
                ImGui::TreePop();
            }
#endif
            ImGui::Separator();
            if(ImGui::Button("Initialize")) {
                // TODO reinitialize pipeline or create a new analyzer window
            }
            ImGui::End();

            // **********************
            // * Superpixel Analyzer window
            // **********************
            static int pointer_x = 0, pointer_y = 0;
            static unsigned int superpixel_id = 0;
            static bool use_spotlight = true;
            static bool use_magnifier = false;

            ImGui::Begin("Superpixel Analyzer");

            cam.capture >> frame;
            cv::cvtColor(frame, frame_rgb, cv::COLOR_BGR2RGB);
#ifdef HAS_LIBGSLIC
            ISuperpixel* superpixel = _superpixel.Compute(frame);
#else
            ISuperpixel* superpixel = _superpixel.Compute(frame);
#endif

            superpixel->GetContour(superpixel_contour);
            superpixel->GetLabels(superpixel_labels);
            superpixel_id = superpixel_labels.at<unsigned int>(pointer_y, pointer_x);
            superpixel_selected = superpixel_labels == superpixel_id;
            cv::meanStdDev(frame_rgb, sel_mean, sel_std, superpixel_selected);
            if (use_spotlight) spotlight(frame_rgb, superpixel_selected, 0.5);
            frame_rgb.setTo(cv::Scalar(200, 5, 240), superpixel_contour);
            
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
#ifndef HAS_LIBGSLIC // with OpenCV SLIC, it is possible to use different superpixel sizes over time
            ImGui::SliderFloat("Superpixel Size", &_superpixel.superpixel_size, 15.0f, 80.0f);
#endif
            ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / ImGui::GetIO().Framerate, ImGui::GetIO().Framerate);
            ImGui::Checkbox("Spotlight", &use_spotlight); ImGui::SameLine(120);
            ImGui::Checkbox("Magnifier", &use_magnifier);

            ImGui::Separator();
            if (ImGui::TreeNode("RGB Histogram")) {
                static bool normalize_component = true;
                ImGui::Checkbox("Normalize Superpixel", &normalize_component);
                RGBHistogram hist(frame, imHistogram.width, imHistogram.height, (use_spotlight?0.3f:1.0f), normalize_component);
                hist.Compute(histogram, use_spotlight?superpixel_selected:cv::noArray());
                imHistogram.Load(histogram.data);
                ImGui::Image(imHistogram.id(), imHistogram.size(), ImVec2(0,0), ImVec2(1,1), ImVec4(1.0f,1.0f,1.0f,1.0f), ImVec4(1.0f,1.0f,1.0f,0.5f));
                ImGui::TreePop();
            }
            ImGui::End();
        }
        app.Render(clear_color);
    }
    app.Shutdown();
    return 0;
}
