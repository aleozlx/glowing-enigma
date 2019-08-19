#include "dcnn.hpp"
#ifdef HAS_TF
namespace tf = tensorflow;

#define MODEL_WEIGHTS "/tank/datasets/research/model_weights/"

TensorFlowInference::TensorFlowInference(std::string const &graph_path) {
    tf::Status status = tf::ReadBinaryProto(tf::Env::Default(), graph_path, &graph);
    if ((this->_loaded = status.ok())) {
        std::cerr << "Graph loaded " << graph_path << std::endl;
    }
    else {
        std::cerr << "WARNING Cannot load graph " << graph_path << std::endl;
    }
}

TensorFlowInference::~TensorFlowInference() {
    if (session)
        session->Close();
}

void TensorFlowInference::Summary() {
    if (!_loaded) return;
    std::cout << "Model Summary" << std::endl;
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
        else {
            std::cout << "[ " << graph.node(i).name() << " ]" << std::endl;
        }
    }
    std::cout << "EOF Model Summary" << std::endl;
}

bool TensorFlowInference::NewSession() {
    if (!_loaded) return false;
    tf::Status status = tf::NewSession(tf::SessionOptions(), &session);
    if (!status.ok()) return false;
    status = session->Create(graph);
    return status.ok();
}


VGG16::VGG16():
    TensorFlowInference(MODEL_WEIGHTS "vgg16.frozen.pb")
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

VGG16SP::VGG16SP():
    TensorFlowInference(MODEL_WEIGHTS "vgg16sp.frozen.pb")
{
    // TODO fix this
    SetInputResolution(256, 256);
}

void VGG16SP::SetInputResolution(unsigned int width, unsigned int height) {
    this->input_shape = tf::TensorShape({1, height, width, 3});
    this->input_tensor = tf::Tensor(tf::DT_UINT8, input_shape);
    this->superpixel_shape = tf::TensorShape({1, height, width});
    this->superpixel_tensor = tf::Tensor(tf::DT_INT32, superpixel_shape);
    this->inputs = {
        { "DataSource/input_image:0", input_tensor },
        { "DataSource/input_superpixels:0", superpixel_tensor },
    };
}

void VGG16SP::Compute(cv::InputArray frame, cv::InputArray superpixels) {
    if(!session) return;

    cv::Mat image, _superpixels;
    // image.convertTo(image, CV_32FC3);
    cv::resize(frame, image, cv::Size(256, 256));
    // std::cout<<"resize1"<<std::endl;
    // TODO it is difficult to efficiently resize an int array, fixing at 256x256 for now
    // CV_Assert(superpixels.type() == CV_32SC1);
    // std::cout<<"superpixel input size "<< superpixels.cols() <<" "<< superpixels.rows() <<std::endl;
    // cv::resize(superpixels, _superpixels, cv::Size(256, 256), 0, 0, cv::INTER_LINEAR);
    // std::cout<<"resize2"<<std::endl;
    _superpixels = superpixels.getMat();
    
    // * DOUBLE CHECK: SIZE TYPE CONTINUITY
    CV_Assert(image.type() == CV_8UC3);
    tf::StringPiece input_buffer = input_tensor.tensor_data();
    std::memcpy(const_cast<char*>(input_buffer.data()), image.data, input_shape.num_elements() * sizeof(char));
    CV_Assert(_superpixels.type() == CV_32SC1);
    tf::StringPiece superpixel_buffer = superpixel_tensor.tensor_data();
    std::memcpy(const_cast<char*>(superpixel_buffer.data()), _superpixels.data, superpixel_shape.num_elements() * sizeof(int));

    tf::Status status = session->Run(inputs, {"Superpixels/MatMul:0"}, {}, &outputs);
    if (!status.ok()) {
        std::cout << status.ToString() << "\n";
        return;
    }

    // Output has shape [S, K]
    // std::cout << outputs[0].DebugString() << "\n";
}

int VGG16SP::GetFeatureDim() const {
    return 512;
}

int VGG16SP::GetNSP() const {
    return 300;
}

void VGG16SP::GetFeature(int superpixel_id, float *output_array) const {
    // Make a copy to detach the lifetime of the output feature array from the tensor.
    
    // std::cerr<<"tensor ptr "<<(const void*)outputs[0].flat<float>().data()<<" superpixel "<<superpixel_id<<std::endl;
    const float *input_array = outputs[0].flat<float>().data() + superpixel_id * GetFeatureDim();
    // std::cerr<<"memcpy "<<(const void*) input_array<<" >> "<<(void*)output_array<<std::endl;
    std::memcpy(output_array, input_array, GetFeatureDim() * sizeof(float));
}

#endif
