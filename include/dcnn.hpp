#ifndef __DCNN_HPP__
#define __DCNN_HPP__

#ifdef HAS_TF
#include <string>
#include <opencv2/core/utility.hpp>
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

    virtual void Compute(cv::InputArray frame, cv::OutputArray output) = 0;

    protected:
    bool _loaded;
};

#endif // HAS_TF

#endif
