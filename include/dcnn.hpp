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

class TensorFlowInference {
    public:
    tensorflow::GraphDef graph;
    tensorflow::Session *session = nullptr;

    TensorFlowInference(std::string const &graph_path);
    virtual ~TensorFlowInference();
    virtual bool NewSession();
    virtual void Summary();

    virtual void Compute(cv::InputArray frame) {}

    protected:
    bool _loaded;
};

class VGG16: public TensorFlowInference {
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

#endif // HAS_TF

#endif
