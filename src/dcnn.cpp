#include "dcnn.hpp"
#ifdef HAS_TF
namespace tf = tensorflow;

TensorFlowInference::TensorFlowInference(std::string const &graph_path) {
    tf::Status status = tf::ReadBinaryProto(tf::Env::Default(), graph_path, &graph);
    this->_loaded = status.ok();
}

TensorFlowInference::~TensorFlowInference() {
    if (session)
        session->Close();
}

void TensorFlowInference::Summary() {
    if (!_loaded) return;

    for (int i = 0; i < graph.node_size(); i++) {
        auto const &node = graph.node(i);
        auto const &attr_map = node.attr();
        std::vector<int64_t> tensor_shape;
        auto const &shape_attr = attr_map.find("shape");
        if (shape_attr != attr_map.end()) {
            auto const &value = shape_attr->second;
            auto const &shape = value.shape();
            for (int i=0; i<shape.dim_size(); ++i) {
                auto const &dim = shape.dim(i);
                auto const &dim_size = dim.size();
                tensor_shape.push_back(dim_size);
            }
            std::cout << "[ ";
            std::copy(tensor_shape.begin(), tensor_shape.end(), std::ostream_iterator<int>(std::cout, " "));
            auto const &dtype_attr = attr_map.find("dtype");
            if (dtype_attr != attr_map.end()) {
                std::cout << tf::DataTypeString(dtype_attr->second.type()) << ' ';
            }
            std::cout << graph.node(i).name() << " ]" << std::endl;
        }           
    }
}

bool TensorFlowInference::NewSession() {
    if (!_loaded) return false;
    tf::Status status = tf::NewSession(tf::SessionOptions(), &session);
    if (!status.ok()) return false;
    status = session->Create(graph);
    return status.ok();
}


VGG16::VGG16():
    TensorFlowInference("/tank/datasets/research/model_weights/vgg16spo.frozen.pb")
{
    // TODO fix this
    SetInputResolution(256, 256);
}

void VGG16::SetInputResolution(unsigned int width, unsigned int height) {
    this->input_shape = tf::TensorShape({1, height, width, 3});
    this->input_tensor = tf::Tensor(tf::DT_UINT8, input_shape);
    this->inputs = {
        { "DataSource/Placeholder:0", input_tensor },
    };
}

void VGG16::Compute(cv::InputArray frame) {
    if(!session) return;

    cv::Mat image;
    // image.convertTo(image, CV_32FC3);
    cv::resize(frame, image, cv::Size(256, 256));
    
    // * DOUBLE CHECK: SIZE TYPE CONTINUITY
    CV_Assert(image.type() == CV_8UC3);
    tf::StringPiece input_buffer = input_tensor.tensor_data();
    std::memcpy(const_cast<char*>(input_buffer.data()), image.data, input_shape.num_elements() * sizeof(char));

    tf::Status status = session->Run(inputs, {"DCNN/block5_pool/MaxPool:0"}, {}, &outputs);
    if (!status.ok()) {
        std::cout << status.ToString() << "\n";
        return;
    }

    // https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/tensor.h
    // auto output_c = outputs[0].scalar<float>();
    
    // // Print the results
    // std::cout << outputs[0].DebugString() << "\n"; // Tensor<type: float shape: [] values: 30>
    // std::cout << output_c() << "\n"; // 30
}
#endif
