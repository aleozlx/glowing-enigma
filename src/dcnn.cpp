#include "dcnn.hpp"
#ifdef HAS_TF

namespace spt::dnn {
    namespace tf = tensorflow;

#include "build_vars.hpp"

    TensorFlowInference::TensorFlowInference(std::string const &graph_path) {
        tf::Status status = tf::ReadBinaryProto(tf::Env::Default(), graph_path, &graph);
        if ((this->_loaded = status.ok())) {
            std::cerr << "Graph loaded " << graph_path << std::endl;
        } else {
            std::cerr << "WARNING Cannot load graph " << graph_path << std::endl;
        }
    }

    TensorFlowInference::~TensorFlowInference() {
        if (session) {
            tf::Status status = session->Close();
            if (!status.ok()) {
                std::cerr << "Failed to close a tf::Session." << std::endl;
            }
        }
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
                for (int i = 0; i < shape.dim_size(); ++i) {
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
            } else {
                std::cout << "[ " << graph.node(i).name() << " ]" << std::endl;
            }
        }
        std::cout << "EOF Model Summary" << std::endl;
    }

    bool TensorFlowInference::NewSession() {
        return this->NewSession(tf::SessionOptions());
    }

    bool TensorFlowInference::NewSession(const tf::SessionOptions &config) {
        if (!_loaded) return false;
        tf::Status status = tf::NewSession(config, &session);
        if (!status.ok()) { // TODO more consistent way of logging
            std::cerr<<"tf::NewSession() error:"<<std::endl;
            std::cerr<<status.ToString()<<std::endl;
            return false;
        }
        status = session->Create(graph);
        if (!status.ok()) {
            std::cerr<<"tf::NewSession() error:"<<std::endl;
            std::cerr<<status.ToString()<<std::endl;
            return false;
        }
        return true;
    }


    VGG16::VGG16() :
            TensorFlowInference(MODEL_WEIGHTS "vgg16.frozen.pb") {
        // TODO fix this
        SetInputResolution(256, 256);
    }

    void VGG16::SetInputResolution(unsigned int width, unsigned int height) {
        this->input_shape = tf::TensorShape({1, height, width, 3});
        this->input_tensor = tf::Tensor(tf::DT_UINT8, input_shape);
        this->inputs = {
                {"DataSource/Placeholder:0", input_tensor},
        };
    }

    void VGG16::Compute(cv::InputArray frame) {
        if (!session) return;

        cv::Mat image;
        // image.convertTo(image, CV_32FC3);
        cv::resize(frame, image, cv::Size(256, 256));

        // * DOUBLE CHECK: SIZE TYPE CONTINUITY
        CV_Assert(image.type() == CV_8UC3);
        tf::StringPiece input_buffer = input_tensor.tensor_data();
        std::memcpy(const_cast<char *>(input_buffer.data()), image.data, input_shape.num_elements() * sizeof(char));

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

    VGG16SP::VGG16SP() :
            TensorFlowInference(MODEL_WEIGHTS "vgg16sp.frozen.pb") {
    }

    void VGG16SP::SetInputResolution(unsigned int width, unsigned int height) {
        std::cerr<<"SetInputResolution(): adding input tensors"<<std::endl;
        this->input_shape = tf::TensorShape({1, height, width, 3});
        this->input_tensor = tf::Tensor(tf::DT_UINT8, input_shape);
        this->superpixel_shape = tf::TensorShape({1, height, width});
        this->superpixel_tensor = tf::Tensor(tf::DT_INT32, superpixel_shape);
        this->inputs = {
                {"DataSource/input_image:0",       input_tensor},
                {"DataSource/input_superpixels:0", superpixel_tensor},
        };
    }

    IComputeFrameSuperpixel* VGG16SP::Compute(cv::InputArray frame, cv::InputArray superpixels) {
        if (!session) return nullptr;

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
        auto input_buffer = this->input_tensor.tensor<tf::uint8, 4>();
        auto superpixel_buffer = this->superpixel_tensor.tensor<tf::int32, 3>();
        tf::uint8 *_input_buffer = input_buffer.data();
        tf::int32 *_superpixel_buffer = superpixel_buffer.data();

//        std::cout<<input_tensor.DeviceSafeDebugString()<<std::endl;
//        std::cout<<"VGG16SP::Compute(): copy "<<input_shape.num_elements() * sizeof(char)<<" Bytes from "<< (void*)image.data <<" to "<< (void*)_input_buffer <<std::endl;
//        std::cout<<superpixel_tensor.DeviceSafeDebugString()<<std::endl;
//        std::cout<<"VGG16SP::Compute(): copy "<<superpixel_shape.num_elements() * sizeof(int)<<" Bytes from "<< (void*)_superpixels.data <<" to "<< (void*)_superpixel_buffer <<std::endl;
//        std::cout<<"Tensor is initialized? "<<input_tensor.IsInitialized()<<std::endl;
        std::memcpy(_input_buffer, image.data, input_shape.num_elements() * sizeof(char));

        CV_Assert(_superpixels.type() == CV_32SC1);
        std::memcpy(_superpixel_buffer, _superpixels.data,
                    superpixel_shape.num_elements() * sizeof(int));

        tf::Status status = session->Run(inputs, {"Superpixels/MatMul:0"}, {}, &outputs);
        if (!status.ok()) {
            std::cout << status.ToString() << "\n";
            return nullptr;
        }
        return this;
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

    void VGG16SP::GetFeature(float *output_array) const {
        // Make a copy to detach the lifetime of the output feature array from the tensor.
        const float *input_array = outputs[0].flat<float>().data();
        std::memcpy(output_array, input_array, GetFeatureDim() * GetNSP() * sizeof(float));
    }


/// Calculate maximum number of chips, with size `c`, that can be fit into length `L` with overlapping proportion `e`.
/// The condition to satisfy is that the right side of the last chip still lies within the given length.
/// c+(n-1)(1-e)c <= L  i.e.  n <= (L-c)/((1-e)c)+1
    int Chipping::NChip(int c, int L, float e) {
        if (e == 0.0) return L / c;
        else return (int) (((float) (L - c)) / ((1 - e) * c) + 1);
    }

    Chipping::Chipping(cv::Size input_size, cv::Size chip_size, float overlap) {
        width = input_size.width;
        height = input_size.height;
        chip_width = chip_size.width;
        chip_height = chip_size.height;
        nx = NChip(chip_width, width, overlap);
        ny = NChip(chip_height, height, overlap);
        nchip = nx * ny;
        if (overlap == 0.0) {
            chip_stride_x = chip_width;
            chip_stride_y = chip_height;
        } else {
            chip_stride_x = (int) ((1 - overlap) * chip_width);
            chip_stride_y = (int) ((1 - overlap) * chip_height);
        }
    }

    cv::Rect Chipping::GetROI(int chip_id) {
        int offset_x = chip_id % nx * chip_stride_x;
        int offset_y = chip_id / nx * chip_stride_y;
        return cv::Rect(offset_x, offset_y, chip_width, chip_height);
    }
}

#endif
