#ifndef __MISC_OCV_HPP__
#define __MISC_OCV_HPP__
#include <string>
#include <vector>
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

struct CameraInfo {
    std::string name;
    int id;
    bool isOpened;

    bool Acquire() {
        return isOpened == false ? (isOpened = true) : false;
    }

    void Release() {
        isOpened = false;
    }
};

namespace cv_misc {
std::vector<int> camera_enumerate() {
    std::vector<int> ret;
    cv::VideoCapture cap;
    for (int i = 0; i<4; i++) {
        if(cap.open(i)) {
            ret.push_back(i);
            cap.release();
        }
    }
    return ret;
}

std::vector<CameraInfo> camera_enumerate2() {
    std::vector<CameraInfo> ret;
    cv::VideoCapture cap;
    for (int i = 0; i<4; i++) {
        if(cap.open(i)) {
            char camera_name[20];
            std::snprintf(camera_name, IM_ARRAYSIZE(camera_name), "/dev/video%d", i);
            ret.push_back({ .name = std::string(camera_name), .id = i, .isOpened = false });
            cap.release();
        }
    }
    return ret;
}

std::string type2str(int type) {
    /*
    std::string ty = cv_misc::type2str(frame_rgb.type());
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
}
#endif
