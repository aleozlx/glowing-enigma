#ifndef __MISC_OCV_HPP__
#define __MISC_OCV_HPP__
#include <string>
#include <vector>
#include <opencv2/videoio.hpp>
#include <opencv2/imgproc.hpp>

/// OpenCV Video I/O
struct Camera {
    int capture_id;
    cv::VideoCapture capture;
    unsigned int width, height;
    Camera(unsigned int idx, unsigned int width, unsigned int height);
    int open();
};

struct CameraInfo {
    std::string name;
    int id;
    bool isOpened;
    bool Acquire();
    void Release();
};

namespace cv_misc {
    std::vector<int> camera_enumerate();
    std::vector<CameraInfo> camera_enumerate2();
    std::string type2str(int type);

    namespace fx {
        void spotlight(cv::OutputArray _frame, cv::InputArray _sel, float alpha);

        // TODO investigate ImGui::PlotHistogram
        struct RGBHistogram {
            float alpha;
            unsigned int width, height;
            std::vector<cv::Mat> bgr_planes;
            bool normalize_component;
            RGBHistogram(cv::InputArray inputRGBImage, unsigned int width, unsigned int height, float alpha=1.0f, bool normalize_component=false);
            void Compute(cv::OutputArray output, cv::InputArray mask=cv::noArray());
        };
    }

}
#endif
