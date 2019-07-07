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
