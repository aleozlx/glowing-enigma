#include "superpixel_pipeline.hpp"

ISuperpixel* OpenCVSLIC::Compute() {
    segmentation->iterate(num_iter);
    segmentation->enforceLabelConnectivity(min_size);
    return dynamic_cast<ISuperpixel*>(this);
}

OpenCVSLIC::OpenCVSLIC(cv::Mat frame, float superpixel_size, float ruler, unsigned int num_iter, float min_size) {
    this->superpixel_size = superpixel_size;
    this->ruler = ruler;
    this->num_iter = num_iter;
    this->min_size = min_size;
    this->segmentation = cv::ximgproc::createSuperpixelSLIC(
        frame, cv::ximgproc::SLIC+1, (int)superpixel_size, ruler);
}

void OpenCVSLIC::GetContour(cv::Mat &output) {
    segmentation->getLabelContourMask(output, true);
}

void OpenCVSLIC::GetLabels(cv::Mat &output) {
    segmentation->getLabels(output);
}

unsigned int OpenCVSLIC::GetNumSuperpixels() {
    return segmentation->getNumberOfSuperpixels();
}
