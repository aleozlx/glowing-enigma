#include <iostream>
#include <string>
#include <vector>
#include <tuple>

#include "app.hpp"
#include "teximage.hpp"
#include "misc_ocv.hpp"
#include "misc_os.hpp"
#include "superpixel.hpp"
#include "dcnn.hpp"

#if __has_include(<filesystem>)
#include <filesystem>
namespace fs = std::filesystem;
#else
#include <experimental/filesystem>
namespace fs = std::experimental::filesystem;
#endif

const unsigned int WIDTH = 432;
const unsigned int HEIGHT = 240;

class SuperpixelAnalyzerWindow: public IWindow {
    public:
    SuperpixelAnalyzerWindow(int frame_width, int frame_height, CameraInfo *_camera_info):
        io(ImGui::GetIO()),
        width(frame_width),
        height(frame_height),
        _camera_info(_camera_info),
        cam(_camera_info->id, frame_width, frame_height)
    {
        std::string id = _camera_info->name; //IWindow::uuid(5);
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
        cv::Size frame_size = frame.size();
        width = frame_size.width;
        height = frame_size.height;
        channels = 3;
        imSuperpixels = TexImage(width, height, channels);
        imHistogram = TexImage(width - 20, height, channels);
        #ifdef HAS_LIBGSLIC
        _superpixel = spt::GSLIC({
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
        _superpixel = spt::OpenCVSLIC(32, 30.0f, 3, 10.0f);
        #endif
        dcnn.Summary();
        dcnn.NewSession();
        dcnn.SetInputResolution(256, 256);
        
        this->_is_shown = true;
        return dynamic_cast<IWindow*>(this);
    }

    bool Draw() override {
        if (!this->_is_shown) return false;
        ImGui::Begin(this->_title, &this->_is_shown);
        cam.capture >> frame;
        cv::cvtColor(frame, frame_rgb, cv::COLOR_BGR2RGB);
        frame_dcnn = frame_rgb.clone();
        spt::ISuperpixel* superpixel = _superpixel.Compute(frame);
        superpixel->GetLabels(superpixel_labels);
        superpixel_id = superpixel_labels.at<unsigned int>(pointer_y, pointer_x);
        superpixel_selected = superpixel_labels == superpixel_id;
        switch (sel.mode) {
            case SuperpixelSelection::Mode::None:
            break;

            case SuperpixelSelection::Mode::Spotlight:
            cv_misc::fx::spotlight(frame_rgb, superpixel_selected, 0.5);
            superpixel->GetContour(superpixel_contour);
            frame_rgb.setTo(cv::Scalar(200, 5, 240), superpixel_contour);
            break;

            case SuperpixelSelection::Mode::Contour:
            cv::findContours(superpixel_selected, superpixel_sel_contour, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);
            cv::drawContours(frame_rgb, superpixel_sel_contour, 0, cv::Scalar(200, 5, 240), 2);
            break;
        }
        
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
                cv::Scalar sel_mean, sel_std;
                cv::meanStdDev(frame_rgb, sel_mean, sel_std, superpixel_selected);
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
        ImGui::Checkbox("Magnifier", &use_magnifier);
        if (ImGui::RadioButton("None", sel.mode == SuperpixelSelection::Mode::None)) {
            sel.mode = SuperpixelSelection::Mode::None;
        }
        ImGui::SameLine();
        if (ImGui::RadioButton("Spotlight", sel.mode == SuperpixelSelection::Mode::Spotlight)) {
            sel.mode = SuperpixelSelection::Mode::Spotlight;
        }
        ImGui::SameLine();
        if (ImGui::RadioButton("Contour", sel.mode == SuperpixelSelection::Mode::Contour)) {
            sel.mode = SuperpixelSelection::Mode::Contour;
        }

        if (ImGui::TreeNode("RGB Histogram")) {
            ImGui::Checkbox("Normalize Superpixel", &normalize_component);
            cv_misc::fx::RGBHistogram hist(frame, imHistogram.width, imHistogram.height, (sel?0.3f:1.0f), normalize_component);
            hist.Compute(histogram_rgb, sel?superpixel_selected:cv::noArray());
            imHistogram.Load(histogram_rgb.data);
            ImGui::Image(imHistogram.id(), imHistogram.size(), ImVec2(0,0), ImVec2(1,1), ImVec4(1.0f,1.0f,1.0f,1.0f), ImVec4(1.0f,1.0f,1.0f,0.5f));
            ImGui::TreePop();
        }

        if (sel.mode == SuperpixelSelection::Mode::Contour && ImGui::TreeNode("Moments")) {
            superpixel_moments = cv::moments(superpixel_sel_contour[0], true);
            // spatial moments
            //   m00, m10, m01, m20, m11, m02, m30, m21, m12, m03;
            // central moments
            //   mu20, mu11, mu02, mu30, mu21, mu12, mu03;
            // central normalized moments
            //   nu20, nu11, nu02, nu30, nu21, nu12, nu03;
            const double v0 = superpixel_moments.m00,
                    v1 = superpixel_moments.mu02,
                    v2 = superpixel_moments.mu20,
                    v3 = superpixel_moments.mu11;
            ImGui::Text("Area: %.1f", v0);
            ImGui::Text("Centroid: (%4.1f,%4.1f)", superpixel_moments.m10/v0, superpixel_moments.m01/v0);
            ImGui::Text("Covariance: [%4.1f %4.1f; . %4.1f]", v2/v0, v3/v0, v1/v0);
            double d = sqrt(v3*v3*4+(v1-v2)*(v1-v2));
            double e1 = (v1+v2+d)/(2*v0), e2 = (v1+v2-d)/(2*v0);
            ImGui::Text("Eigenvalues: (%4.1f,%4.1f)", e1, e2);
            ImGui::Text("Eccentricity: %4.1f", sqrt(1.0-e2/e1));
            double hu[7];
            cv::HuMoments(superpixel_moments, hu);
            ImGui::Text("Hu: %4.1f %4.1f %4.1f %4.1f %4.1f %4.1f %4.1f", hu[0], hu[1], hu[2], hu[3], hu[4], hu[5], hu[6]);
            ImGui::TreePop();
        }

        if (ImGui::TreeNode("Superpixel DCNN Features")) {
            dcnn.Compute(frame_dcnn);
            
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
    spt::GSLIC _superpixel;
    #else
    spt::OpenCVSLIC _superpixel;
    #endif
    spt::dnn::VGG16 dcnn;

    bool _is_shown = false;
    char _title[64];
    int pointer_x = 0, pointer_y = 0;
    unsigned int superpixel_id = 0;
    SuperpixelSelection sel = {SuperpixelSelection::Mode::Contour};
    bool use_magnifier = false;
    bool normalize_component = true;

    cv::Mat frame, frame_rgb, frame_dcnn;
    cv::Mat superpixel_contour, superpixel_labels, superpixel_selected;
    std::vector<std::vector<cv::Point>> superpixel_sel_contour;
    cv::Moments superpixel_moments;
    cv::Mat histogram_rgb;
    
};

template <typename SIZE>
SIZE gdiv(SIZE a, SIZE b) {
    return (a + b - 1) / b;
}

class SuperpixelAnalyzerWindow2: public IWindow {
    protected:
        template<typename T>
        struct LazyLoader {
            T val, _new_val;

            explicit LazyLoader(T init_val): val(init_val) {

            }

            /// Sync value to a temp var for data binding
            T* operator&() { // NOLINT I know exactly what I am doing!
                _new_val = val;
                return &_new_val;
            }

            /// Sync value back and detect changes
            bool Update() {
                return Update(_new_val);
            }

            /// Update value and detect changes
            bool Update(T new_val) {
                if(new_val == val) return false;
                else {
                    val = new_val;
                    return true;
                }
            }
        };
    public:
    SuperpixelAnalyzerWindow2(int frame_width, int frame_height, std::string glob_pattern, float superpixel_size, float chip_overlap, bool dcnn_enable=true):
        io(ImGui::GetIO()),
        width(frame_width),
        height(frame_height),
        glob_dataset(glob_pattern.c_str()),
        d_image_id(0),
        superpixel_size(superpixel_size),
        chip_overlap(chip_overlap),
        dcnn_enable(dcnn_enable)
    {
        std::string id = "xView";
        std::snprintf(_title, IM_ARRAYSIZE(_title), "Superpixel Analyzer [%s]", id.c_str());
        io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;  // Enable Keyboard Controls
    }

    IWindow* Show() override {
        if(glob_dataset.size()<=0)
            return nullptr;

        //"/tank/datasets/research/xView/train_images/1036.tif";
        this->ReinitializeRawFrame();
        frame = frame_raw(chips.GetROI(0));

        cv::Size frame_size = frame.size();
        width = frame_size.width;
        height = frame_size.height;
        channels = 3;
        std::cout<<"Superpixel processing size "<<width<<"x"<<height<<std::endl;
        imSuperpixels = TexImage(width, height, channels);
        imHistogram = TexImage(width - 20, height, channels);
        #ifdef HAS_LIBGSLIC
        _superpixel = spt::GSLIC({
            .img_size = { width, height },
            .no_segs = 64,
            .spixel_size = (int)superpixel_size,
            .no_iters = 5,
            .coh_weight = 0.6f,
            .do_enforce_connectivity = true,
            .color_space = gSLICr::CIELAB, // gSLICr::XYZ | gSLICr::RGB
            .seg_method = gSLICr::GIVEN_SIZE // gSLICr::GIVEN_NUM
        });
        #else
        _superpixel = spt::OpenCVSLIC(superpixel_size, 30.0f, 3, 10.0f);
        #endif

        if (dcnn_enable) {
            dcnn.Summary();
            dcnn.NewSession();
            dcnn.SetInputResolution(256, 256);
        }
        this->_is_shown = true;
        return dynamic_cast<IWindow*>(this);
    }

    void ReinitializeRawFrame() {
        frame_raw = cv::imread(glob_dataset[d_image_id.val], cv::IMREAD_COLOR);
        cv::Size real_size = frame_raw.size();
        chips = spt::dnn::Chipping(real_size, cv::Size(width, height), chip_overlap);
        d_chip_id = 0;
    }

    bool Draw() override {
        if (!this->_is_shown) return false;
        ImGui::Begin(this->_title, &this->_is_shown);
        ImGui::SliderInt("Image id", &d_image_id, 0, static_cast<int>(glob_dataset.size())-1);
        if(d_image_id.Update())
            this->ReinitializeRawFrame();
        ImGui::Text("File name: %s", glob_dataset[d_image_id.val]);
        ImGui::SliderInt("Chip id", &d_chip_id, 0, chips.nchip-1);
        frame = frame_raw(chips.GetROI(d_chip_id));
        this->superpixel = _superpixel.Compute(frame);
        superpixel->GetLabels(superpixel_labels);
        cv::cvtColor(frame, frame_rgb, cv::COLOR_BGR2RGB);
        frame_dcnn = frame_rgb.clone();

        superpixel_id = superpixel_labels.at<unsigned int>(pointer_y, pointer_x);
        superpixel_selected = superpixel_labels == superpixel_id;
        switch (sel.mode) {
            case SuperpixelSelection::Mode::None:
            break;

            case SuperpixelSelection::Mode::Spotlight:
            cv_misc::fx::spotlight(frame_rgb, superpixel_selected, 0.5);
            superpixel->GetContour(superpixel_contour);
            frame_rgb.setTo(cv::Scalar(200, 5, 240), superpixel_contour);
            break;

            case SuperpixelSelection::Mode::Contour:
            cv::findContours(superpixel_selected, superpixel_sel_contour, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);
            cv::drawContours(frame_rgb, superpixel_sel_contour, 0, cv::Scalar(200, 5, 240), 2);
            break;
        }
        
        imSuperpixels.Load(frame_rgb.data);
        ImGui::Text("(%d, %d) => (%d,)", imSuperpixels.width, imSuperpixels.height, superpixel->GetNumSuperpixels());
        // superpixel_id bound check
        if(superpixel_id>=superpixel->GetNumSuperpixels()) {
            // ! BUG
            std::cerr<<"Resetting superpixel_id because it is out of bound"<<std::endl;
            superpixel_id = 0;
        }
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
                cv::Scalar sel_mean, sel_std;
                cv::meanStdDev(frame_rgb, sel_mean, sel_std, superpixel_selected);
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
        ImGui::Checkbox("Magnifier", &use_magnifier);
        if (ImGui::RadioButton("None", sel.mode == SuperpixelSelection::Mode::None)) {
            sel.mode = SuperpixelSelection::Mode::None;
        }
        ImGui::SameLine();
        if (ImGui::RadioButton("Spotlight", sel.mode == SuperpixelSelection::Mode::Spotlight)) {
            sel.mode = SuperpixelSelection::Mode::Spotlight;
        }
        ImGui::SameLine();
        if (ImGui::RadioButton("Contour", sel.mode == SuperpixelSelection::Mode::Contour)) {
            sel.mode = SuperpixelSelection::Mode::Contour;
        }

        if (ImGui::TreeNode("RGB Histogram")) {
            ImGui::Checkbox("Normalize Superpixel", &normalize_component);
            cv_misc::fx::RGBHistogram hist(frame, imHistogram.width, imHistogram.height, (sel?0.3f:1.0f), normalize_component);
            hist.Compute(histogram_rgb, sel?superpixel_selected:cv::noArray());
            imHistogram.Load(histogram_rgb.data);
            ImGui::Image(imHistogram.id(), imHistogram.size(), ImVec2(0,0), ImVec2(1,1), ImVec4(1.0f,1.0f,1.0f,1.0f), ImVec4(1.0f,1.0f,1.0f,0.5f));
            ImGui::TreePop();
        }

        if (sel.mode == SuperpixelSelection::Mode::Contour && ImGui::TreeNode("Moments")) {
            superpixel_moments = cv::moments(superpixel_sel_contour[0], true);
            // spatial moments
            //   m00, m10, m01, m20, m11, m02, m30, m21, m12, m03;
            // central moments
            //   mu20, mu11, mu02, mu30, mu21, mu12, mu03;
            // central normalized moments
            //   nu20, nu11, nu02, nu30, nu21, nu12, nu03;
            const double v0 = superpixel_moments.m00,
                    v1 = superpixel_moments.mu02,
                    v2 = superpixel_moments.mu20,
                    v3 = superpixel_moments.mu11;
            ImGui::Text("Area: %.1f", v0);
            ImGui::Text("Centroid: (%4.1f,%4.1f)", superpixel_moments.m10/v0, superpixel_moments.m01/v0);
            ImGui::Text("Covariance: [%4.1f %4.1f; . %4.1f]", v2/v0, v3/v0, v1/v0);
            double d = sqrt(v3*v3*4+(v1-v2)*(v1-v2));
            double e1 = (v1+v2+d)/(2*v0), e2 = (v1+v2-d)/(2*v0);
            ImGui::Text("Eigenvalues: (%4.1f,%4.1f)", e1, e2);
            ImGui::Text("Eccentricity: %4.1f", sqrt(1.0-e2/e1));
            double hu[7];
            cv::HuMoments(superpixel_moments, hu);
            ImGui::Text("Hu: %4.1f %4.1f %4.1f %4.1f %4.1f %4.1f %4.1f", hu[0], hu[1], hu[2], hu[3], hu[4], hu[5], hu[6]);
            ImGui::TreePop();
        }

        if (dcnn_enable && ImGui::TreeNode("Superpixel DCNN Features")) {
            dcnn.Compute(frame_dcnn, superpixel_labels);
            superpixel_feature_buffer.resize(dcnn.GetFeatureDim());
            dcnn.GetFeature(superpixel_id, superpixel_feature_buffer.data());
            ImGui::PlotLines("Feature", superpixel_feature_buffer.data(), dcnn.GetFeatureDim(), 0, "", -1.0f, 1.0f, ImVec2(320, 200));
            ImGui::TreePop();
        }

        ImGui::End();
        return true;
    }

    protected:
    ImGuiIO& io;
    int width, height, channels;
    os_misc::Glob glob_dataset;
    TexImage imSuperpixels;
    TexImage imHistogram;
    #ifdef HAS_LIBGSLIC
    spt::GSLIC _superpixel;
    #else
    spt::OpenCVSLIC _superpixel;
    #endif
    spt::ISuperpixel* superpixel;
    spt::dnn::VGG16SP dcnn;

    bool _is_shown = false;
    char _title[64];
    int pointer_x = 0, pointer_y = 0;
    LazyLoader<int> d_image_id;
    float superpixel_size;
    float chip_overlap;
    unsigned int superpixel_id = 0;
    SuperpixelSelection sel = {SuperpixelSelection::Mode::Contour};
    bool use_magnifier = false;
    bool normalize_component = true;
    int d_chip_id = 0;
    spt::dnn::Chipping chips;
    const bool dcnn_enable;

    cv::Mat frame_raw, frame, frame_rgb, frame_dcnn;
    cv::Mat superpixel_contour, superpixel_labels, superpixel_selected;
    std::vector<std::vector<cv::Point>> superpixel_sel_contour;
    cv::Moments superpixel_moments;
    cv::Mat histogram_rgb;
    std::vector<float> superpixel_feature_buffer;
};

std::vector<CameraInfo> cameras;
static std::list<std::unique_ptr<IWindow>> windows;

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
        if(ImGui::TreeNode("Video Feed")) {
            if (cameras.size()>0) {
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
            }
            else ImGui::Text("No cameras found.");
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

std::vector<std::tuple<std::string, std::string>> datasets;

class DatasetWindow: public IStaticWindow {
    public:
    bool Draw() override {
        ImGui::Begin("Dataset");
        static size_t d_dataset = 0;
        if(ImGui::TreeNode("Dataset") && datasets.size()>0) {
            if (ImGui::BeginCombo("Source", std::get<0>(datasets[d_dataset]).c_str())) {
                for (size_t i = 0; i<datasets.size(); ++i) {
                    bool is_selected = i == d_dataset;
                    if (ImGui::Selectable(std::get<0>(datasets[i]).c_str(), is_selected))
                        d_dataset = i;
                    if (is_selected)
                        ImGui::SetItemDefaultFocus();
                }
                ImGui::EndCombo();
            }
            ImGui::TreePop();
        }

        #ifdef HAS_LIBGSLIC // with gSLIC, it is not efficient to use different superpixel sizes over time
        if(ImGui::TreeNode("Superpixels")) { 
            ImGui::SliderFloat("Superpixel Size", &d_superpixel_size, 5.6f, 24.0f);
            ImGui::TreePop();
        }
        #endif

        if(ImGui::TreeNode("DCNN")) {
            ImGui::Checkbox("Enable", &d_dcnn_enable);
            if(d_dcnn_enable) d_chip_size = 256;
            ImGui::TreePop();
        }

        if(!d_dcnn_enable && ImGui::TreeNode("Chipping")) {
            ImGui::SliderInt("sqrt(Chip Size)", &d_chip_size, 100, 1000);
            ImGui::SliderFloat("Overlap percentage", &d_chip_overlap, 0, 0.99);
            ImGui::TreePop();
        }

        if(ImGui::TreeNode("Labels")) {
            ImGui::Checkbox("Enable", &d_label_enable);
            ImGui::TreePop();
        }

        ImGui::Separator();

        if(ImGui::Button("Initialize")) {
            const auto frame_width = d_chip_size;
            const auto frame_height = d_chip_size;
            auto w = std::make_unique<SuperpixelAnalyzerWindow2>(
                frame_width, frame_height,
                std::get<1>(datasets[d_dataset]),
                d_superpixel_size,
                d_chip_overlap,
                d_dcnn_enable);
            if (w->Show() != nullptr)
                windows.push_back(std::move(w));
        }
        ImGui::End();
        return true;
    }

protected:
    float d_superpixel_size = 8.0f;
    bool d_dcnn_enable = false;
    int d_chip_size = 500;
    float d_chip_overlap = 0.3;
    bool d_label_enable = false;
};

#if 0
class MomentsWindow: public IWindow {
    public:
    MomentsWindow(int frame_width, int frame_height, CameraInfo *_camera_info):
        io(ImGui::GetIO()),
        width(frame_width),
        height(frame_height),
        _camera_info(_camera_info),
        cam(_camera_info->id, frame_width, frame_height)
    {
        std::string id = _camera_info->name;
        std::snprintf(_title, IM_ARRAYSIZE(_title), "Superpixel Analyzer [%s]", id.c_str());
        io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;  // Enable Keyboard Controls
    }

    IWindow* Show() override {
        if (!cam.open()) {
            // Cannot receive video feed
            this->_is_shown = false;
            return nullptr;
        }
        cam.capture >> frame;
        cv::Size frame_size = frame.size();
        width = frame_size.width;
        height = frame_size.height;
        channels = 3;

        
        this->_is_shown = true;
        return dynamic_cast<IWindow*>(this);
    }

    bool Draw() override {
        if (!this->_is_shown) return false;
        ImGui::Begin(this->_title, &this->_is_shown);
        cam.capture >> frame;

        ImGui::End();
        return true;
    }

    protected:
    ImGuiIO& io;
    int width, height, channels;
    CameraInfo *_camera_info;
    Camera cam;

    bool _is_shown = false;
    char _title[64];

    cv::Mat frame;
};

class MomentsExperiment: public IStaticWindow {
    public:
    bool Draw() override {
        ImGui::Begin("Moments Experiment");
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

        ImGui::Separator();

        if(ImGui::Button("Initialize")) {
            if (d_camera_current->Acquire()) {
                auto w = std::make_unique<MomentsWindow>(WIDTH, HEIGHT, d_camera_current);
                if (w->Show() != nullptr)
                    windows.push_back(std::move(w));
            }
        }
        ImGui::End();
        return true;
    }
};

// Our vertices. Three consecutive floats give a 3D vertex; Three consecutive vertices give a triangle.
// A cube has 6 faces with 2 triangles each, so this makes 6*2=12 triangles, and 12*3 vertices
static const GLfloat g_vertex_buffer_data[] = {
    -1.0f,-1.0f,-1.0f, // triangle 1 : begin
    -1.0f,-1.0f, 1.0f,
    -1.0f, 1.0f, 1.0f, // triangle 1 : end
    1.0f, 1.0f,-1.0f, // triangle 2 : begin
    -1.0f,-1.0f,-1.0f,
    -1.0f, 1.0f,-1.0f, // triangle 2 : end
    1.0f,-1.0f, 1.0f,
    -1.0f,-1.0f,-1.0f,
    1.0f,-1.0f,-1.0f,
    1.0f, 1.0f,-1.0f,
    1.0f,-1.0f,-1.0f,
    -1.0f,-1.0f,-1.0f,
    -1.0f,-1.0f,-1.0f,
    -1.0f, 1.0f, 1.0f,
    -1.0f, 1.0f,-1.0f,
    1.0f,-1.0f, 1.0f,
    -1.0f,-1.0f, 1.0f,
    -1.0f,-1.0f,-1.0f,
    -1.0f, 1.0f, 1.0f,
    -1.0f,-1.0f, 1.0f,
    1.0f,-1.0f, 1.0f,
    1.0f, 1.0f, 1.0f,
    1.0f,-1.0f,-1.0f,
    1.0f, 1.0f,-1.0f,
    1.0f,-1.0f,-1.0f,
    1.0f, 1.0f, 1.0f,
    1.0f,-1.0f, 1.0f,
    1.0f, 1.0f, 1.0f,
    1.0f, 1.0f,-1.0f,
    -1.0f, 1.0f,-1.0f,
    1.0f, 1.0f, 1.0f,
    -1.0f, 1.0f,-1.0f,
    -1.0f, 1.0f, 1.0f,
    1.0f, 1.0f, 1.0f,
    -1.0f, 1.0f, 1.0f,
    1.0f,-1.0f, 1.0f
};

// One color for each vertex. They were generated randomly.
static const GLfloat g_color_buffer_data[] = {
    0.583f,  0.771f,  0.014f,
    0.609f,  0.115f,  0.436f,
    0.327f,  0.483f,  0.844f,
    0.822f,  0.569f,  0.201f,
    0.435f,  0.602f,  0.223f,
    0.310f,  0.747f,  0.185f,
    0.597f,  0.770f,  0.761f,
    0.559f,  0.436f,  0.730f,
    0.359f,  0.583f,  0.152f,
    0.483f,  0.596f,  0.789f,
    0.559f,  0.861f,  0.639f,
    0.195f,  0.548f,  0.859f,
    0.014f,  0.184f,  0.576f,
    0.771f,  0.328f,  0.970f,
    0.406f,  0.615f,  0.116f,
    0.676f,  0.977f,  0.133f,
    0.971f,  0.572f,  0.833f,
    0.140f,  0.616f,  0.489f,
    0.997f,  0.513f,  0.064f,
    0.945f,  0.719f,  0.592f,
    0.543f,  0.021f,  0.978f,
    0.279f,  0.317f,  0.505f,
    0.167f,  0.620f,  0.077f,
    0.347f,  0.857f,  0.137f,
    0.055f,  0.953f,  0.042f,
    0.714f,  0.505f,  0.345f,
    0.783f,  0.290f,  0.734f,
    0.722f,  0.645f,  0.174f,
    0.302f,  0.455f,  0.848f,
    0.225f,  0.587f,  0.040f,
    0.517f,  0.713f,  0.338f,
    0.053f,  0.959f,  0.120f,
    0.393f,  0.621f,  0.362f,
    0.673f,  0.211f,  0.457f,
    0.820f,  0.883f,  0.371f,
    0.982f,  0.099f,  0.879f
};

class CustomRenderTestWindow: public TestWindow {
    public:
    CustomRenderTestWindow(): TestWindow() {
        
    }

    ~CustomRenderTestWindow() {
        
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
        glGenFramebuffers(1, &buffer);
        
        // The texture we're going to render to
        glGenTextures(1, &renderedTexture);

        // "Bind" the newly created texture : all future texture functions will modify this texture
        glBindTexture(GL_TEXTURE_2D, renderedTexture);

        // Give an empty image to OpenGL ( the last "0" )
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, 640, 480, 0,GL_RGB, GL_UNSIGNED_BYTE, 0);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);

        // Set "renderedTexture" as our colour attachement #0
        glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, renderedTexture, 0);

        // Set the list of draw buffers.
        GLenum DrawBuffers[1] = {GL_COLOR_ATTACHMENT0};
        glDrawBuffers(1, DrawBuffers); // "1" is the size of DrawBuffers

        GLuint colorbuffer;
        glGenBuffers(1, &colorbuffer);
        glBindBuffer(GL_ARRAY_BUFFER, colorbuffer);
        glBufferData(GL_ARRAY_BUFFER, sizeof(g_color_buffer_data), g_color_buffer_data, GL_STATIC_DRAW);
        glEnableVertexAttribArray(1);
        glBindBuffer(GL_ARRAY_BUFFER, colorbuffer);
        glVertexAttribPointer(
            1,                                // attribute. No particular reason for 1, but must match the layout in the shader.
            3,                                // size
            GL_FLOAT,                         // type
            GL_FALSE,                         // normalized?
            0,                                // stride
            (void*)0                          // array buffer offset
        );

        // Render into texture
        glBindFramebuffer(GL_FRAMEBUFFER, buffer);
        glViewport(0,0,640,480);

        glDrawArrays(GL_TRIANGLES, 0, 12*3);
        glBufferData(GL_ARRAY_BUFFER, sizeof(g_color_buffer_data), g_color_buffer_data, GL_STATIC_DRAW);

        // Always check that our framebuffer is ok
        if(glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
            this->_is_shown = true;
            return dynamic_cast<IWindow*>(this);
        }
        else {
            this->_is_shown = false;
            return nullptr;
        }
    }

    protected:
    GLuint buffer = 0;
    GLuint renderedTexture = 0;

    

        //     if (ImGui::TreeNode("Custom Render Test")) {
            
        //     ImGui::TreePop();
        // }
};
#endif

const fs::path pth_xView("/tank/datasets/research/xView");

int main(int, char**) {
    App app = App::Initialize();
    if (!app.ok) return 1;
    ImVec4 clear_color = ImVec4(0.45f, 0.55f, 0.60f, 1.00f);
    cameras = cv_misc::camera_enumerate2();
    datasets.emplace_back("xView", (pth_xView / "train_images/*.tif").string());
//    windows.push_back(std::make_unique<PipelineSettingsWindow>());
    windows.push_back(std::make_unique<DatasetWindow>());

    while (app.EventLoop()){
        for (auto w = windows.begin(); w != windows.end();) {
            if (!(*w)->Draw()) windows.erase(w++);
            else ++w;
        }
        app.Render(clear_color);
    }

    // Ensure dtors are invoked: RAII style CUDA resources
    windows.clear();
    return 0;
}
