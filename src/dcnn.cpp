#include "dcnn.hpp"

namespace tf = tensorflow;

TensorFlowInference::TensorFlowInference(std::string const &graph_path) {
    tf::Status status = tf::ReadBinaryProto(tf::Env::Default(), graph_path, &graph);
    this->_loaded = status.ok();
}

TensorFlowInference::~TensorFlowInference() {
    if (session)
        session->Close();
}

bool TensorFlowInference::NewSession() {
    if (!_loaded) return false;
    tf::Status status = tf::NewSession(tf::SessionOptions(), &session);
    if (!status.ok()) return false;

    // print a summary of the loaded graph
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

    status = session->Create(graph);
    return status.ok();
}
