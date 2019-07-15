#include "superpixel_pipeline.hpp"

ISuperpixel* OpenCVSLIC::Compute(cv::InputArray frame) {
    cv::medianBlur(frame, frame_hsv, 5);
    cv::cvtColor(frame_hsv, frame_hsv, cv::COLOR_BGR2HSV);
    segmentation = cv::ximgproc::createSuperpixelSLIC(
        frame_hsv, cv::ximgproc::SLIC+1, (int)superpixel_size, ruler);
    segmentation->iterate(num_iter);
    segmentation->enforceLabelConnectivity(min_size);
    return dynamic_cast<ISuperpixel*>(this);
}

OpenCVSLIC::OpenCVSLIC(float superpixel_size, float ruler, unsigned int num_iter, float min_size) {
    this->superpixel_size = superpixel_size;
    this->ruler = ruler;
    this->num_iter = num_iter;
    this->min_size = min_size;
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
GSLIC::GSLIC(): 
    in_img(nullptr),
    gSLICr_engine(nullptr)
{

}

GSLIC::GSLIC(gSLICr::objects::settings settings):
    in_img(std::make_unique<gSLICr::UChar4Image>(settings.img_size, true, true)),
    gSLICr_engine(std::make_unique<gSLICr::engines::core_engine>(settings))
{
    this->width = settings.img_size.x;
    this->height = settings.img_size.y;
}

ISuperpixel* GSLIC::Compute(cv::InputArray _frame) {
    cv::Mat frame = _frame.getMat();
    CV_Assert(frame.cols == (int)width && frame.rows == (int)height);
    copy_image(frame, in_img.get());
    gSLICr_engine->Process_Frame(in_img.get());
    return dynamic_cast<ISuperpixel*>(this);
}

void GSLIC::GetContour(cv::OutputArray output) {
    cv::Mat outmat;
    outmat.create(cv::Size(width, height), CV_8UC1);
    gSLICr::MaskImage out_img({(int)width, (int)height}, false, true);
    out_img.use_data_cpu(outmat.data);
    gSLICr_engine->Draw_Boundary_Mask(&out_img);
    out_img.force_download();
    // copy_image(&out_img, outmat); // saved a copy by having OpenCV own data.
    output.assign(outmat);
}

void GSLIC::GetLabels(cv::OutputArray output) {
    cv::Mat outmat;
    outmat.create(cv::Size(width, height), CV_32SC1);
    const gSLICr::IntImage *segmentation = gSLICr_engine->Get_Seg_Res();
    copy_image_c1(segmentation, outmat);
    output.assign(outmat);
    if (actual_num_superpixels==0) { // compute num of superpixels while we are at it
        double m = 0;
        cv::minMaxIdx(outmat, nullptr, &m);
        actual_num_superpixels = static_cast<unsigned int>(m) + 1;
    }
}

unsigned int GSLIC::GetNumSuperpixels() {
    if (actual_num_superpixels==0) {
        const gSLICr::IntImage *segmentation = gSLICr_engine->Get_Seg_Res();
        // a quick & dirty way to get max
        actual_num_superpixels = static_cast<unsigned int>(max_c1(segmentation)) + 1;
    }
    return actual_num_superpixels;
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
