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

#endif
