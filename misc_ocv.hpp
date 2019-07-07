#ifndef __MISC_OCV_HPP__
#define __MISC_OCV_HPP__
#include <string>
#include <opencv2/videoio.hpp>

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
#endif
