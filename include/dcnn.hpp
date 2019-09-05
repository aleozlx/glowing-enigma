#ifndef __DCNN_HPP__
#define __DCNN_HPP__

#ifdef HAS_TF
#include <string>
#include <memory>
#include <cstdlib>
#include <opencv2/core/utility.hpp>
#include <opencv2/imgproc.hpp>
#include <tensorflow/core/public/session.h>
#include <tensorflow/core/platform/env.h>

namespace spt::dnn {
    class TensorFlowInference {
    public:
        tensorflow::GraphDef graph;
        tensorflow::Session *session = nullptr;

        TensorFlowInference(std::string const &graph_path);

        virtual ~TensorFlowInference();

        virtual bool NewSession();

        virtual void Summary();

        // virtual void Compute(cv::InputArray frame) {}

    protected:
        bool _loaded;
    };

    class IComputeFrame {
    public:
        virtual void Compute(cv::InputArray frame) = 0;
    };

    class IComputeFrameSuperpixel {
    public:
        virtual void Compute(cv::InputArray frame, cv::InputArray superpixels) = 0;
    };

    struct Chipping {
        int width, height;
        int chip_width, chip_height;
        int nx, ny, nchip;
        int chip_stride_x, chip_stride_y;

        Chipping() {}

        /// Calculate maximum number of chips, with size `c`, that can be fit into length `L` with overlapping proportion `e`.
        static int NChip(int c, int L, float e = 0.0);

        Chipping(cv::Size input_size, cv::Size chip_size, float overlap = 0.0);

        cv::Rect GetROI(int chip_id);
    };

    class VGG16 : public TensorFlowInference, public IComputeFrame {
    public:
        VGG16();

        void SetInputResolution(unsigned int width, unsigned int height);

        void Compute(cv::InputArray frame) override;

    protected:
        tensorflow::TensorShape input_shape;
        tensorflow::Tensor input_tensor;
        std::vector<std::pair<std::string, tensorflow::Tensor>> inputs;
        std::vector<tensorflow::Tensor> outputs;
    };

    class VGG16SP : public TensorFlowInference, public IComputeFrameSuperpixel {
    public:
        VGG16SP();

        void SetInputResolution(unsigned int width, unsigned int height);

        void Compute(cv::InputArray frame, cv::InputArray superpixels) override;

        int GetFeatureDim() const;

        int GetNSP() const;

        void GetFeature(int superpixel_id, float *output_array) const;

    protected:
        tensorflow::TensorShape input_shape;
        tensorflow::Tensor input_tensor;
        tensorflow::TensorShape superpixel_shape;
        tensorflow::Tensor superpixel_tensor;
        std::vector<std::pair<std::string, tensorflow::Tensor>> inputs;
        std::vector<tensorflow::Tensor> outputs;
    };
}

#endif // HAS_TF

#endif
