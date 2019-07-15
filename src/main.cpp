#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include <list>
#include <cstdlib>

#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/core/utility.hpp>

#include "app.hpp"
#include "teximage.hpp"
#include "superpixel_pipeline.hpp"
#include "misc_ocv.hpp"

#ifdef HAS_TF
#include <tensorflow/core/public/session.h>
#include <tensorflow/core/platform/env.h>
namespace tf = tensorflow;
#endif

void spotlight(cv::OutputArray _frame, cv::InputArray _sel, float alpha) {
    cv::Mat frame = _frame.getMat(), selection = _sel.getMat();
    CV_Assert(frame.type() == CV_8UC3 && selection.type() == CV_8UC1);
    const int channels = 3;
    for (int i=0; i<frame.rows; ++i) {
        unsigned char* fptr = frame.ptr(i);
        const unsigned char* sptr = selection.ptr(i);
        for (int j=0; j<frame.cols; ++j) {
            if (sptr[j]==0) {
                fptr[j*channels] = static_cast<unsigned char>(fptr[j*channels] * alpha);
                fptr[j*channels+1] = static_cast<unsigned char>(fptr[j*channels+1] * alpha);
                fptr[j*channels+2] = static_cast<unsigned char>(fptr[j*channels+2] * alpha);
            }
        }
    }
}

// TODO investigate ImGui::PlotHistogram
struct RGBHistogram {
    float alpha;
    unsigned int width, height;
    std::vector<cv::Mat> bgr_planes;
    bool normalize_component;

    RGBHistogram(cv::InputArray inputRGBImage, unsigned int width, unsigned int height, float alpha=1.0f, bool normalize_component=false) {
        this->width = width;
        this->height = height;
        this->alpha = alpha;
        this->normalize_component = normalize_component;
        cv::split(inputRGBImage, bgr_planes);
    }

    void Compute(cv::OutputArray output, cv::InputArray mask=cv::noArray()) {
        const int histSize = 256;
        const float range[] = { 0, 256 } ;
        const float* histRange = { range };
        bool uniform = true; bool accumulate = false;
        cv::Mat b_hist, g_hist, r_hist;
        cv::calcHist(&bgr_planes[0], 1, 0, cv::noArray(), b_hist, 1, &histSize, &histRange, uniform, accumulate);
        cv::calcHist(&bgr_planes[1], 1, 0, cv::noArray(), g_hist, 1, &histSize, &histRange, uniform, accumulate);
        cv::calcHist(&bgr_planes[2], 1, 0, cv::noArray(), r_hist, 1, &histSize, &histRange, uniform, accumulate);

        // Draw the global histograms for B, G and R
        float bin_w = (double)width / histSize;
        cv::Mat histImage(height, width, CV_8UC3, cv::Scalar(0,0,0));

        // Normalize the result
        double mb = cv::norm(b_hist, cv::NORM_INF),
            mg = cv::norm(g_hist, cv::NORM_INF),
            mr = cv::norm(r_hist, cv::NORM_INF);
        double max_count = std::max({mb, mg, mr});
        b_hist = b_hist / max_count * histImage.rows;
        g_hist = g_hist / max_count * histImage.rows;
        r_hist = r_hist / max_count * histImage.rows;
        
        // Draw for each channel
        // Note: Blue and Red channels are purposefully swapped to avoid color conversion before rendering
        float _alpha = mask.empty()?1.0f:this->alpha;
        for(int i = 1;i < histSize;i++){
            cv::line( histImage, cv::Point( bin_w*(i-1), height - cvRound(b_hist.at<float>(i-1)) ) ,
                            cv::Point( bin_w*(i), height - cvRound(b_hist.at<float>(i)) ),
                            cv::Scalar( 0, 0, 255*_alpha), 2, 8, 0  );
            cv::line( histImage, cv::Point( bin_w*(i-1), height - cvRound(g_hist.at<float>(i-1)) ) ,
                            cv::Point( bin_w*(i), height - cvRound(g_hist.at<float>(i)) ),
                            cv::Scalar( 0, 255*_alpha, 0), 2, 8, 0  );
            cv::line( histImage, cv::Point( bin_w*(i-1), height - cvRound(r_hist.at<float>(i-1)) ) ,
                            cv::Point( bin_w*(i), height - cvRound(r_hist.at<float>(i)) ),
                            cv::Scalar( 255*_alpha, 0, 0), 2, 8, 0  );
        }

        if (!mask.empty()) {
            cv::calcHist(&bgr_planes[0], 1, 0, mask, b_hist, 1, &histSize, &histRange, uniform, accumulate);
            cv::calcHist(&bgr_planes[1], 1, 0, mask, g_hist, 1, &histSize, &histRange, uniform, accumulate);
            cv::calcHist(&bgr_planes[2], 1, 0, mask, r_hist, 1, &histSize, &histRange, uniform, accumulate);
            if (normalize_component) {
                cv::normalize(b_hist, b_hist, 0, histImage.rows, cv::NORM_MINMAX, -1, cv::noArray());
                cv::normalize(g_hist, g_hist, 0, histImage.rows, cv::NORM_MINMAX, -1, cv::noArray());
                cv::normalize(r_hist, r_hist, 0, histImage.rows, cv::NORM_MINMAX, -1, cv::noArray());
            }
            else {
                b_hist = b_hist / max_count * histImage.rows;
                g_hist = g_hist / max_count * histImage.rows;
                r_hist = r_hist / max_count * histImage.rows;
            }
            for(int i = 1;i < histSize;i++){
                cv::line( histImage, cv::Point( bin_w*(i-1), height - cvRound(b_hist.at<float>(i-1)) ) ,
                                cv::Point( bin_w*(i), height - cvRound(b_hist.at<float>(i)) ),
                                cv::Scalar( 0, 0, 255), 2, 8, 0  );
                cv::line( histImage, cv::Point( bin_w*(i-1), height - cvRound(g_hist.at<float>(i-1)) ) ,
                                cv::Point( bin_w*(i), height - cvRound(g_hist.at<float>(i)) ),
                                cv::Scalar( 0, 255, 0), 2, 8, 0  );
                cv::line( histImage, cv::Point( bin_w*(i-1), height - cvRound(r_hist.at<float>(i-1)) ) ,
                                cv::Point( bin_w*(i), height - cvRound(r_hist.at<float>(i)) ),
                                cv::Scalar( 255, 0, 0), 2, 8, 0  );
            }
        }
        output.assign(histImage);
    }
};

class DCNNInference {
    public:
    tf::GraphDef graph;
    tf::Session *session = nullptr;

    DCNNInference(std::string const &graph_path) {
        tf::Status status = tf::ReadBinaryProto(tf::Env::Default(), graph_path, &graph);
        this->_loaded = status.ok();
    }

    virtual ~DCNNInference() {
        if (session)
            session->Close();
    }

    bool NewSession() {
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

    virtual void Compute() = 0;

    // void Compute() {
        // tf::GraphDef graph_def;
        // // status = tf::ReadBinaryProto(tf::Env::Default(), "/tank/datasets/research/model_weights/vgg16.frozen.pb", &graph_def);
        // if (!status.ok()) {
        //     std::cerr << status.ToString() << "\n";
        //     return;
        // }

        // tf::Tensor a(tf::DT_FLOAT, tf::TensorShape());
        // a.scalar<float>()() = 3.0;

        // tf::Tensor b(tf::DT_FLOAT, tf::TensorShape());
        // b.scalar<float>()() = 2.0;

        // std::vector<std::pair<std::string, tensorflow::Tensor>> inputs = {
        //     { "a", a },
        //     { "b", b },
        // };

        // // The session will initialize the outputs
        // std::vector<tf::Tensor> outputs;

        // // Run the session, evaluating our "c" operation from the graph
        // status = session->Run(inputs, {"c"}, {}, &outputs);
        // if (!status.ok()) {
        //     std::cout << status.ToString() << "\n";
        //     return;
        // }

        // // Grab the first output (we only evaluated one graph node: "c")
        // // and convert the node to a scalar representation.
        // auto output_c = outputs[0].scalar<float>();

        // // (There are similar methods for vectors and matrices here:
        // // https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/public/tensor.h)

        // // Print the results
        // std::cout << outputs[0].DebugString() << "\n"; // Tensor<type: float shape: [] values: 30>
        // std::cout << output_c() << "\n"; // 30
    // }

    protected:
    bool _loaded;
};

class VGG16: public DCNNInference {
    public:
    VGG16(): DCNNInference("/tank/datasets/research/model_weights/vgg16.frozen.pb") {

    }

    void Compute() override {

    }
};

const unsigned int WIDTH = 432;
const unsigned int HEIGHT = 240;

class IWindow {
    public:
    virtual IWindow* Show() = 0;
    virtual bool Draw() = 0;
    virtual ~IWindow() {}

    static std::string uuid(const int len) {
        static const char alphanum[] =
            "0123456789"
            "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
            "abcdefghijklmnopqrstuvwxyz";
        char buf[len+1];
        buf[len] = '\0';
        for (int i = 0; i < len; ++i)
            buf[i] = alphanum[rand() % (sizeof(alphanum) - 1)];
        return std::string(buf);
    }
};

class TestWindow: public IWindow {
    public:
    TestWindow(bool show = false) {
        std::string id = IWindow::uuid(5);
        std::snprintf(_title, IM_ARRAYSIZE(_title), "Test Window [%s]", id.c_str());
        this->_is_shown = show;
    }

    bool Draw() override {
        if (!this->_is_shown) return false;
        ImGui::Begin(this->_title, &this->_is_shown);
        ImGui::Text("Hello from another window!");
        if (ImGui::Button("Close Me"))
            this->_is_shown = false;
        ImGui::End();
        return true;
    }

    IWindow* Show() override {
        this->_is_shown = true;
        return dynamic_cast<IWindow*>(this);
    }

    protected:
    bool _is_shown = false;
    char _title[32];
};

class SuperpixelAnalyzerWindow: public IWindow {
    public:
    SuperpixelAnalyzerWindow(int frame_width, int frame_height, CameraInfo *_camera_info):
        io(ImGui::GetIO()),
        width(frame_width),
        height(frame_height),
        _camera_info(_camera_info),
        cam(_camera_info->id, frame_width, frame_height)
    {
        std::string id = IWindow::uuid(5);
        std::snprintf(_title, IM_ARRAYSIZE(_title), "Superpixel Analyzer [%s]", id.c_str());
        io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;  // Enable Keyboard Controls
    }

    ~SuperpixelAnalyzerWindow() {
        _camera_info->Release();
    }

    IWindow* Show() override {
        if (!cam.open()) {
            // Cannot receive video feed
            this->_is_shown = false;
            return nullptr;
        }
        cam.capture >> frame;
        frame_size = frame.size();
        width = frame_size.width;
        height = frame_size.height;
        channels = 3;
        imSuperpixels = TexImage(width, height, channels);
        imHistogram = TexImage(width - 20, height, channels);
#ifdef HAS_LIBGSLIC
        _superpixel = GSLIC({
            .img_size = { width, height },
            .no_segs = 64,
            .spixel_size = 32,
            .no_iters = 5,
            .coh_weight = 0.6f,
            .do_enforce_connectivity = true,
            .color_space = gSLICr::XYZ, // gSLICr::CIELAB | gSLICr::RGB
            .seg_method = gSLICr::GIVEN_SIZE // gSLICr::GIVEN_NUM
        });
#else
        _superpixel = OpenCVSLIC(32, 30.0f, 3, 10.0f);
#endif
        this->_is_shown = true;
        return dynamic_cast<IWindow*>(this);
    }

    bool Draw() override {
        if (!this->_is_shown) return false;
        ImGui::Begin(this->_title, &this->_is_shown);
        cam.capture >> frame;
        cv::cvtColor(frame, frame_rgb, cv::COLOR_BGR2RGB);
#ifdef HAS_LIBGSLIC
        ISuperpixel* superpixel = _superpixel.Compute(frame);
#else
        ISuperpixel* superpixel = _superpixel.Compute(frame);
#endif

        superpixel->GetContour(superpixel_contour);
        superpixel->GetLabels(superpixel_labels);
        superpixel_id = superpixel_labels.at<unsigned int>(pointer_y, pointer_x);
        superpixel_selected = superpixel_labels == superpixel_id;
        cv::meanStdDev(frame_rgb, sel_mean, sel_std, superpixel_selected);
        if (use_spotlight) spotlight(frame_rgb, superpixel_selected, 0.5);
        frame_rgb.setTo(cv::Scalar(200, 5, 240), superpixel_contour);
        
        imSuperpixels.Load(frame_rgb.data);
        ImGui::Text("(%d, %d) => (%d,)", imSuperpixels.width, imSuperpixels.height, superpixel->GetNumSuperpixels());
        ImVec2 pos = ImGui::GetCursorScreenPos();
        ImGui::Image(imSuperpixels.id(), imSuperpixels.size(), ImVec2(0,0), ImVec2(1,1), ImVec4(1.0f,1.0f,1.0f,1.0f), ImVec4(1.0f,1.0f,1.0f,0.5f));
        if (ImGui::IsItemHovered()) {
            pointer_x = static_cast<int>(io.MousePos.x - pos.x);
            pointer_y = static_cast<int>(io.MousePos.y - pos.y);
            if (use_magnifier) {
                ImGui::BeginTooltip();
                float region_sz = 32.0f;
                float region_x = io.MousePos.x - pos.x - region_sz * 0.5f; if (region_x < 0.0f) region_x = 0.0f; else if (region_x > imSuperpixels.f32width - region_sz) region_x = imSuperpixels.f32width - region_sz;
                float region_y = io.MousePos.y - pos.y - region_sz * 0.5f; if (region_y < 0.0f) region_y = 0.0f; else if (region_y > imSuperpixels.f32height - region_sz) region_y = imSuperpixels.f32height - region_sz;
                float zoom = 4.0f;
                ImGui::Text("Ptr: (%d,%d) Id: %d", pointer_x, pointer_y, superpixel_id);
                ImGui::Text("Mean: (%.1f,%.1f,%.1f)", sel_mean[0], sel_mean[1], sel_mean[2]);
                ImGui::Text("Std: (%.1f,%.1f,%.1f)", sel_std[0], sel_std[1], sel_std[2]);
                ImGui::Image(
                    imSuperpixels.id(), ImVec2(region_sz * zoom, region_sz * zoom),
                    imSuperpixels.uv(region_x, region_y), imSuperpixels.uv(region_x + region_sz, region_y + region_sz),
                    ImVec4(1.0f, 1.0f, 1.0f, 1.0f), ImVec4(1.0f, 1.0f, 1.0f, 0.5f));
                ImGui::EndTooltip();
            }
        }
#ifndef HAS_LIBGSLIC // with OpenCV SLIC, it is possible to use different superpixel sizes over time
        ImGui::SliderFloat("Superpixel Size", &_superpixel.superpixel_size, 15.0f, 80.0f);
#endif
        ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / ImGui::GetIO().Framerate, ImGui::GetIO().Framerate);
        ImGui::Checkbox("Spotlight", &use_spotlight); ImGui::SameLine(120);
        ImGui::Checkbox("Magnifier", &use_magnifier);

        if (ImGui::TreeNode("RGB Histogram")) {
            ImGui::Checkbox("Normalize Superpixel", &normalize_component);
            RGBHistogram hist(frame, imHistogram.width, imHistogram.height, (use_spotlight?0.3f:1.0f), normalize_component);
            hist.Compute(histogram, use_spotlight?superpixel_selected:cv::noArray());
            imHistogram.Load(histogram.data);
            ImGui::Image(imHistogram.id(), imHistogram.size(), ImVec2(0,0), ImVec2(1,1), ImVec4(1.0f,1.0f,1.0f,1.0f), ImVec4(1.0f,1.0f,1.0f,0.5f));
            ImGui::TreePop();
        }

        if (ImGui::TreeNode("Superpixel DCNN Features")) {
            ImGui::Text("Test.");
            ImGui::TreePop();
        }
        ImGui::End();
        return true;
    }

    protected:
    ImGuiIO& io;
    int width, height, channels;
    CameraInfo *_camera_info;
    Camera cam;
    TexImage imSuperpixels;
    TexImage imHistogram;
#ifdef HAS_LIBGSLIC
    GSLIC _superpixel;
#else
    OpenCVSLIC _superpixel;
#endif

    bool _is_shown = false;
    char _title[32];
    int pointer_x = 0, pointer_y = 0;
    unsigned int superpixel_id = 0;
    bool use_spotlight = true;
    bool use_magnifier = false;
    bool normalize_component = true;

    cv::Mat frame, frame_rgb;
    cv::Mat superpixel_contour, superpixel_labels, superpixel_selected;
    cv::Scalar sel_mean, sel_std;
    cv::Mat histogram, histogram_rgb;
    cv::Size frame_size;
};

class IStaticWindow: public IWindow {
    public:
    virtual IWindow* Show() override {
        return dynamic_cast<IWindow*>(this);
    }
    virtual ~IStaticWindow() {}
};

std::vector<CameraInfo> cameras;
std::list<std::unique_ptr<IWindow>> windows;

class PipelineSettingsWindow: public IStaticWindow {
    public:
    bool Draw() override {
        ImGui::Begin("Pipeline Settings");
        if(ImGui::TreeNode("Enabled Features")) {
            #define FEATURE(FEATURE_MACRO) ImGui::Text("-D " #FEATURE_MACRO)
            #define FEATURE_VER(FEATURE_MACRO) ImGui::Text("-D " #FEATURE_MACRO " %s", VERSTR(FEATURE_MACRO))
            #include "features.hpp"
            ImGui::TreePop();
        }

        static CameraInfo *d_camera_current = &cameras[0];
        if(ImGui::TreeNode("Video Feed") && cameras.size()>0) {
            if (ImGui::BeginCombo("Source", d_camera_current->name.c_str())) {
                for (auto &camera_info: cameras) {
                    bool is_selected = (d_camera_current == &camera_info);
                    if (ImGui::Selectable(camera_info.name.c_str(), is_selected))
                        d_camera_current = &camera_info;
                    if (is_selected)
                        ImGui::SetItemDefaultFocus();
                }
                ImGui::EndCombo();
            }
            ImGui::TreePop();
        }

#ifdef HAS_LIBGSLIC // with gSLIC, it is not efficient to use different superpixel sizes over time
        if(ImGui::TreeNode("Superpixels")) {
            static float d_superpixel_size = 32.0f;
            ImGui::SliderFloat("Superpixel Size", &d_superpixel_size, 15.0f, 80.0f);
            ImGui::TreePop();
        }
#endif
        ImGui::Separator();

        if(ImGui::Button("Initialize")) {
            if (d_camera_current->Acquire()) {
                auto w = std::make_unique<SuperpixelAnalyzerWindow>(WIDTH, HEIGHT, d_camera_current);
                if (w->Show() != nullptr)
                    windows.push_back(std::move(w));
            }
        }
        ImGui::End();
        return true;
    }
};


int main(int, char**) {
    App app = App::Initialize();
    if (!app.ok) return 1;
    GLFWwindow* window = app.window;
    ImVec4 clear_color = ImVec4(0.45f, 0.55f, 0.60f, 1.00f);
    cameras = cv_misc::camera_enumerate2();
    windows.push_back(std::make_unique<PipelineSettingsWindow>());

    VGG16 dcnn;
    dcnn.NewSession();

    while (!glfwWindowShouldClose(window)){
        glfwPollEvents();
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();
        for (auto w = windows.begin(); w != windows.end();) {
            if (!(*w)->Draw()) windows.erase(w++);
            else ++w;
        }
        app.Render(clear_color);
    }
    app.Shutdown();
    return 0;
}
