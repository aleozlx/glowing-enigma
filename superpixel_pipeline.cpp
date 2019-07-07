#include "superpixel_pipeline.hpp"

ISuperpixel* OpenCVSLIC::Compute() {
    segmentation->iterate(num_iter);
    segmentation->enforceLabelConnectivity(min_size);
    return dynamic_cast<ISuperpixel*>(this);
}

OpenCVSLIC::OpenCVSLIC(cv::InputArray frame, float superpixel_size, float ruler, unsigned int num_iter, float min_size) {
    this->superpixel_size = superpixel_size;
    this->ruler = ruler;
    this->num_iter = num_iter;
    this->min_size = min_size;
    this->segmentation = cv::ximgproc::createSuperpixelSLIC(
        frame, cv::ximgproc::SLIC+1, (int)superpixel_size, ruler);
}

void OpenCVSLIC::GetContour(cv::OutputArray output) {
    segmentation->getLabelContourMask(output, true);
}

void OpenCVSLIC::GetLabels(cv::OutputArray output) {
    segmentation->getLabels(output);
}

unsigned int OpenCVSLIC::GetNumSuperpixels() {
    return segmentation->getNumberOfSuperpixels();
}

#ifdef HAS_LIBGSLIC
GSLIC::GSLIC(gSLICr::objects::settings settings):
    in_img(settings.img_size, true, true),
    out_img(settings.img_size, true, true),
    gSLICr_engine(settings)
{
    this->width = settings.img_size.x;
    this->height = settings.img_size.y;
}

void GSLIC::with(cv::InputArray frame) {
    this->frame = frame.getMat();
}

ISuperpixel* GSLIC::Compute() {
    CV_Assert(frame.cols == (int)width && frame.rows == (int)height);
    copy_image(frame, &in_img);
    gSLICr_engine.Process_Frame(&in_img);
    return dynamic_cast<ISuperpixel*>(this);
}
// #include <iostream>
void GSLIC::GetContour(cv::OutputArray output) {
    cv::Mat outmat;
    outmat.create(cv::Size(width, height), CV_8UC3);
    // std::cout<<"Draw()"<<std::endl;
    gSLICr_engine.Draw_Segmentation_Result(&out_img);
    // std::cout<<"Draw()e"<<std::endl;
    copy_image(&out_img, outmat);
    output.assign(outmat);
}

void GSLIC::GetLabels(cv::OutputArray output) {

}

unsigned int GSLIC::GetNumSuperpixels() {
    return 0;
}

void GSLIC::copy_image(const cv::Mat& inimg, gSLICr::UChar4Image* outimg) {
	gSLICr::Vector4u* outimg_ptr = outimg->GetData(MEMORYDEVICE_CPU);

	for (int y = 0; y < outimg->noDims.y;y++) {
        const unsigned char* iptr = inimg.ptr(y);
		for (int x = 0; x < outimg->noDims.x; x++)
		{
			int idx = x + y * outimg->noDims.x;
			outimg_ptr[idx].b = iptr[x*3];
			outimg_ptr[idx].g = iptr[x*3+1];
			outimg_ptr[idx].r = iptr[x*3+2];
		}
    }
}

void GSLIC::copy_image(const gSLICr::UChar4Image* inimg, cv::Mat& outimg) {
	const gSLICr::Vector4u* inimg_ptr = inimg->GetData(MEMORYDEVICE_CPU);

	for (int y = 0; y < inimg->noDims.y; y++) {
        unsigned char* optr = outimg.ptr(y);
		for (int x = 0; x < inimg->noDims.x; x++)
		{
			int idx = x + y * inimg->noDims.x;
            // Note: output is RGB
			optr[x*3] = inimg_ptr[idx].r;
			optr[x*3+1] = inimg_ptr[idx].g;
			optr[x*3+2] = inimg_ptr[idx].b;
		}
    }
}
#endif
