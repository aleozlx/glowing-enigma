#ifndef __SUPERPIXEL_PIPELINE_HPP__
#define __SUPERPIXEL_PIPELINE_HPP__
#include <opencv2/core/utility.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/ximgproc.hpp>

class ISuperpixel {
    public:
    virtual ISuperpixel* Compute(cv::InputArray frame) = 0;
    virtual void GetContour(cv::OutputArray output) = 0;
    virtual void GetLabels(cv::OutputArray output) = 0;
    virtual unsigned int GetNumSuperpixels() = 0;
    virtual ~ISuperpixel() {}
};

class OpenCVSLIC: public ISuperpixel {
    public:
    OpenCVSLIC(float superpixel_size, float ruler, unsigned int num_iter, float min_size);
    ISuperpixel* Compute(cv::InputArray frame) override;
    void GetContour(cv::OutputArray output) override;
    void GetLabels(cv::OutputArray output) override;
    unsigned int GetNumSuperpixels() override;

    unsigned int num_iter;
    float superpixel_size, min_size, ruler;
    cv::Ptr<cv::ximgproc::SuperpixelSLIC> segmentation;

    protected:
    cv::Mat frame_hsv;
};

#ifdef HAS_LIBGSLIC
#include "gSLICr_Lib/gSLICr.h"
class GSLIC: public ISuperpixel {
    public:
    GSLIC(gSLICr::objects::settings settings);
    ISuperpixel* Compute(cv::InputArray frame) override;
    void GetContour(cv::OutputArray output) override;
    void GetLabels(cv::OutputArray output) override;
    unsigned int GetNumSuperpixels() override;

    protected:
    unsigned int width, height;
    unsigned int actual_num_superpixels = 0;
    gSLICr::UChar4Image in_img;
    gSLICr::engines::core_engine gSLICr_engine;
    static void copy_image(const cv::Mat& inimg, gSLICr::UChar4Image* outimg);
    static void copy_image(const gSLICr::UChar4Image* inimg, cv::Mat& outimg);
    
    template <typename T>
    static void copy_image_c1(const ORUtils::Image<T>* inimg, cv::Mat& outimg){
        const T* inimg_ptr = inimg->GetData(MEMORYDEVICE_CPU);
        for (int y = 0; y < inimg->noDims.y; y++) {
            T* optr = reinterpret_cast<T*>(outimg.ptr(y));
            for (int x = 0; x < inimg->noDims.x; x++) {
                int idx = x + y * inimg->noDims.x;
                optr[x] = inimg_ptr[idx];
            }
        }
    }

    template <typename T>
    static T max_c1(const ORUtils::Image<T>* inimg) {
        int n = inimg->noDims.x * inimg->noDims.y;
        const T* inimg_ptr = inimg->GetData(MEMORYDEVICE_CPU);
        T _m = 0;
        for (int p = 0; p < n; p++) {
            int d = inimg_ptr[p];
            if (d>_m) _m = d;
        }
        return _m;
    }
};
#endif

#endif
