#include <iostream>
#include "misc_ocv.hpp"

using namespace std;
using namespace cv;

namespace cv_misc {
    Camera::Camera(unsigned int idx, unsigned int width, unsigned int height) : capture_id(idx), capture(idx) {
        this->width = width;
        this->height = height;
    }

    int Camera::open() {
        if (!capture.open(capture_id)) {
            cerr << "Cannot initialize video:" << capture_id << endl;
            return 0;
        } else {
            capture.set(CAP_PROP_FRAME_WIDTH, width);
            capture.set(CAP_PROP_FRAME_HEIGHT, height);
            return 1;
        }
    }

    bool CameraInfo::Acquire() {
        return isOpened == false ? (isOpened = true) : false;
    }

    void CameraInfo::Release() {
        isOpened = false;
    }

    vector<int> camera_enumerate() {
        vector<int> ret;
        VideoCapture cap;
        for (int i = 0; i < 4; i++) {
            if (cap.open(i)) {
                ret.push_back(i);
                cap.release();
            }
        }
        return ret;
    }

    vector<CameraInfo> camera_enumerate2() {
        vector<CameraInfo> ret;
        VideoCapture cap;
        for (int i = 0; i < 4; i++) {
            if (cap.open(i)) {
                const int BUFSZ = 20;
                char camera_name[BUFSZ];
                snprintf(camera_name, BUFSZ, "/dev/video%d", i);
                ret.push_back({.name = string(camera_name), .id = i, .isOpened = false});
                cap.release();
            }
        }
        return ret;
    }

    string type2str(int type) {
        /*
        string ty = cv_misc::type2str(frame_rgb.type());
        ImGui::Text("type %s", ty.c_str());
        */
        string r;
        unsigned char depth = type & CV_MAT_DEPTH_MASK;
        unsigned char chans = 1 + (type >> CV_CN_SHIFT);
        switch (depth) {
            case CV_8U:
                r = "8U";
                break;
            case CV_8S:
                r = "8S";
                break;
            case CV_16U:
                r = "16U";
                break;
            case CV_16S:
                r = "16S";
                break;
            case CV_32S:
                r = "32S";
                break;
            case CV_32F:
                r = "32F";
                break;
            case CV_64F:
                r = "64F";
                break;
            default:
                r = "User";
                break;
        }
        r += "C";
        r += (chans + '0');
        return r;
    }

    void fx::spotlight(OutputArray _frame, InputArray _sel, float alpha) {
        Mat frame = _frame.getMat(), selection = _sel.getMat();
        CV_Assert(frame.type() == CV_8UC3 && selection.type() == CV_8UC1);
        const int channels = 3;
        for (int i = 0; i < frame.rows; ++i) {
            unsigned char *fptr = frame.ptr(i);
            const unsigned char *sptr = selection.ptr(i);
            for (int j = 0; j < frame.cols; ++j) {
                if (sptr[j] == 0) {
                    fptr[j * channels] = static_cast<unsigned char>(fptr[j * channels] * alpha);
                    fptr[j * channels + 1] = static_cast<unsigned char>(fptr[j * channels + 1] * alpha);
                    fptr[j * channels + 2] = static_cast<unsigned char>(fptr[j * channels + 2] * alpha);
                }
            }
        }
    }

    fx::RGBHistogram::RGBHistogram(InputArray frame, unsigned int width, unsigned int height, float alpha,
                                            bool normalize_component) {
        this->width = width;
        this->height = height;
        this->alpha = alpha;
        this->normalize_component = normalize_component;
        split(frame, bgr_planes);
    }

    void fx::RGBHistogram::Compute(OutputArray output, InputArray mask) {
        const int histSize = 256;
        const float range[] = {0, 256};
        const float *histRange = {range};
        bool uniform = true;
        bool accumulate = false;
        Mat b_hist, g_hist, r_hist;
        calcHist(&bgr_planes[0], 1, 0, noArray(), b_hist, 1, &histSize, &histRange, uniform, accumulate);
        calcHist(&bgr_planes[1], 1, 0, noArray(), g_hist, 1, &histSize, &histRange, uniform, accumulate);
        calcHist(&bgr_planes[2], 1, 0, noArray(), r_hist, 1, &histSize, &histRange, uniform, accumulate);

        // Draw the global histograms for B, G and R
        float bin_w = (double) width / histSize;
        Mat histImage(height, width, CV_8UC3, Scalar(0, 0, 0));

        // Normalize the result
        double mb = norm(b_hist, NORM_INF),
                mg = norm(g_hist, NORM_INF),
                mr = norm(r_hist, NORM_INF);
        double max_count = max({mb, mg, mr});
        b_hist = b_hist / max_count * histImage.rows;
        g_hist = g_hist / max_count * histImage.rows;
        r_hist = r_hist / max_count * histImage.rows;

        // Draw for each channel
        // Note: Blue and Red channels are purposefully swapped to avoid color conversion before rendering
        float _alpha = mask.empty() ? 1.0f : this->alpha;
        for (int i = 1; i < histSize; i++) {
            line(histImage, Point(bin_w * (i - 1), height - cvRound(b_hist.at<float>(i - 1))),
                 Point(bin_w * (i), height - cvRound(b_hist.at<float>(i))),
                 Scalar(0, 0, 255 * _alpha), 2, 8, 0);
            line(histImage, Point(bin_w * (i - 1), height - cvRound(g_hist.at<float>(i - 1))),
                 Point(bin_w * (i), height - cvRound(g_hist.at<float>(i))),
                 Scalar(0, 255 * _alpha, 0), 2, 8, 0);
            line(histImage, Point(bin_w * (i - 1), height - cvRound(r_hist.at<float>(i - 1))),
                 Point(bin_w * (i), height - cvRound(r_hist.at<float>(i))),
                 Scalar(255 * _alpha, 0, 0), 2, 8, 0);
        }

        if (!mask.empty()) {
            calcHist(&bgr_planes[0], 1, 0, mask, b_hist, 1, &histSize, &histRange, uniform, accumulate);
            calcHist(&bgr_planes[1], 1, 0, mask, g_hist, 1, &histSize, &histRange, uniform, accumulate);
            calcHist(&bgr_planes[2], 1, 0, mask, r_hist, 1, &histSize, &histRange, uniform, accumulate);
            if (normalize_component) {
                normalize(b_hist, b_hist, 0, histImage.rows, NORM_MINMAX, -1, noArray());
                normalize(g_hist, g_hist, 0, histImage.rows, NORM_MINMAX, -1, noArray());
                normalize(r_hist, r_hist, 0, histImage.rows, NORM_MINMAX, -1, noArray());
            } else {
                b_hist = b_hist / max_count * histImage.rows;
                g_hist = g_hist / max_count * histImage.rows;
                r_hist = r_hist / max_count * histImage.rows;
            }
            for (int i = 1; i < histSize; i++) {
                line(histImage, Point(bin_w * (i - 1), height - cvRound(b_hist.at<float>(i - 1))),
                     Point(bin_w * (i), height - cvRound(b_hist.at<float>(i))),
                     Scalar(0, 0, 255), 2, 8, 0);
                line(histImage, Point(bin_w * (i - 1), height - cvRound(g_hist.at<float>(i - 1))),
                     Point(bin_w * (i), height - cvRound(g_hist.at<float>(i))),
                     Scalar(0, 255, 0), 2, 8, 0);
                line(histImage, Point(bin_w * (i - 1), height - cvRound(r_hist.at<float>(i - 1))),
                     Point(bin_w * (i), height - cvRound(r_hist.at<float>(i))),
                     Scalar(255, 0, 0), 2, 8, 0);
            }
        }
        output.assign(histImage);
    }

    fx::HSVHistogram::HSVHistogram(InputArray frame, unsigned int width, unsigned int height, float alpha,
                                            bool normalize_component) {
        this->width = width;
        this->height = height;
        this->alpha = alpha;
        this->normalize_component = normalize_component;
        Mat frame_hsv;
        cvtColor(frame, frame_hsv, cv::COLOR_BGR2HSV);
        split(frame_hsv, hsv_planes);
    }

    void fx::HSVHistogram::Compute(OutputArray output, InputArray mask) {
        // const int histSize = 256;
        // const float range[] = { 0, 256 } ;
        // const float* histRange = { range };
        // Mat b_hist, g_hist, r_hist;
        // calcHist(&bgr_planes[0], 1, 0, noArray(), b_hist, 1, &histSize, &histRange);
        // calcHist(&bgr_planes[1], 1, 0, noArray(), g_hist, 1, &histSize, &histRange);
        // calcHist(&bgr_planes[2], 1, 0, noArray(), r_hist, 1, &histSize, &histRange);

        // // Draw the global histograms for B, G and R
        // float bin_w = (double)width / histSize;
        // Mat histImage(height, width, CV_8UC3, Scalar(0,0,0));

        // // Normalize the result
        // double mb = norm(b_hist, NORM_INF),
        //     mg = norm(g_hist, NORM_INF),
        //     mr = norm(r_hist, NORM_INF);
        // double max_count = max({mb, mg, mr});
        // b_hist = b_hist / max_count * histImage.rows;
        // g_hist = g_hist / max_count * histImage.rows;
        // r_hist = r_hist / max_count * histImage.rows;

        // // Draw for each channel
        // float _alpha = mask.empty()?1.0f:this->alpha;
        // for(int i = 1;i < histSize;i++){
        //     line( histImage, Point( bin_w*(i-1), height - cvRound(b_hist.at<float>(i-1)) ) ,
        //                     Point( bin_w*(i), height - cvRound(b_hist.at<float>(i)) ),
        //                     Scalar( 0, 0, 255*_alpha), 2, 8, 0  );
        //     line( histImage, Point( bin_w*(i-1), height - cvRound(g_hist.at<float>(i-1)) ) ,
        //                     Point( bin_w*(i), height - cvRound(g_hist.at<float>(i)) ),
        //                     Scalar( 0, 255*_alpha, 0), 2, 8, 0  );
        //     line( histImage, Point( bin_w*(i-1), height - cvRound(r_hist.at<float>(i-1)) ) ,
        //                     Point( bin_w*(i), height - cvRound(r_hist.at<float>(i)) ),
        //                     Scalar( 255*_alpha, 0, 0), 2, 8, 0  );
        // }

        // if (!mask.empty()) {
        //     calcHist(&bgr_planes[0], 1, 0, mask, b_hist, 1, &histSize, &histRange);
        //     calcHist(&bgr_planes[1], 1, 0, mask, g_hist, 1, &histSize, &histRange);
        //     calcHist(&bgr_planes[2], 1, 0, mask, r_hist, 1, &histSize, &histRange);
        //     if (normalize_component) {
        //         normalize(b_hist, b_hist, 0, histImage.rows, NORM_MINMAX, -1, noArray());
        //         normalize(g_hist, g_hist, 0, histImage.rows, NORM_MINMAX, -1, noArray());
        //         normalize(r_hist, r_hist, 0, histImage.rows, NORM_MINMAX, -1, noArray());
        //     }
        //     else {
        //         b_hist = b_hist / max_count * histImage.rows;
        //         g_hist = g_hist / max_count * histImage.rows;
        //         r_hist = r_hist / max_count * histImage.rows;
        //     }
        //     for(int i = 1;i < histSize;i++){
        //         line( histImage, Point( bin_w*(i-1), height - cvRound(b_hist.at<float>(i-1)) ) ,
        //                         Point( bin_w*(i), height - cvRound(b_hist.at<float>(i)) ),
        //                         Scalar( 0, 0, 255), 2, 8, 0  );
        //         line( histImage, Point( bin_w*(i-1), height - cvRound(g_hist.at<float>(i-1)) ) ,
        //                         Point( bin_w*(i), height - cvRound(g_hist.at<float>(i)) ),
        //                         Scalar( 0, 255, 0), 2, 8, 0  );
        //         line( histImage, Point( bin_w*(i-1), height - cvRound(r_hist.at<float>(i-1)) ) ,
        //                         Point( bin_w*(i), height - cvRound(r_hist.at<float>(i)) ),
        //                         Scalar( 255, 0, 0), 2, 8, 0  );
        //     }
        // }
        // output.assign(histImage);
    }
}
