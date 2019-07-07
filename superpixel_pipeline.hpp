#ifndef __SUPERPIXEL_PIPELINE_HPP__
#define __SUPERPIXEL_PIPELINE_HPP__
#include <opencv2/core/utility.hpp>
#include <opencv2/ximgproc.hpp>

class ISuperpixel {
    public:
    virtual ISuperpixel* Compute() = 0;
    virtual void GetContour(cv::OutputArray output) = 0;
    virtual void GetLabels(cv::OutputArray output) = 0;
    virtual unsigned int GetNumSuperpixels() = 0;
};

class OpenCVSLIC: public ISuperpixel {
    public:
    OpenCVSLIC(cv::InputArray frame, float superpixel_size, float ruler, unsigned int num_iter, float min_size);
    ISuperpixel* Compute() override;
    void GetContour(cv::OutputArray output) override;
    void GetLabels(cv::OutputArray output) override;
    unsigned int GetNumSuperpixels() override;

    unsigned int num_iter;
    float superpixel_size, min_size, ruler;
    cv::Ptr<cv::ximgproc::SuperpixelSLIC> segmentation;
};

#ifdef HAS_LIBGSLIC
#include "gSLICr/gSLICr_Lib/gSLICr.h"
class GSLIC: public ISuperpixel {
    public:
    GSLIC(gSLICr::objects::settings settings);
    void with(cv::InputArray frame);
    ISuperpixel* Compute() override;
    void GetContour(cv::OutputArray output) override;
    void GetLabels(cv::OutputArray output) override;
    unsigned int GetNumSuperpixels() override;

    protected:
    cv::Mat frame;
    unsigned int width, height;
    gSLICr::UChar4Image in_img, out_img;
    gSLICr::engines::core_engine gSLICr_engine;
    static void copy_image(const cv::Mat& inimg, gSLICr::UChar4Image* outimg);
    static void copy_image(const gSLICr::UChar4Image* inimg, cv::Mat& outimg);
};
#endif

#endif
