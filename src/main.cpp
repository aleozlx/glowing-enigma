#include <iostream>
#include <string>
#include <vector>

#include "app.hpp"
#include "teximage.hpp"
#include "misc_ocv.hpp"
#include "superpixel.hpp"
#include "dcnn.hpp"

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
        dcnn.Summary();
        dcnn.NewSession();
        
        this->_is_shown = true;
        return dynamic_cast<IWindow*>(this);
    }

    bool Draw() override {
        if (!this->_is_shown) return false;
        ImGui::Begin(this->_title, &this->_is_shown);
        cam.capture >> frame;
        cv::cvtColor(frame, frame_rgb, cv::COLOR_BGR2RGB);
        frame_dcnn = frame_rgb.clone();
#ifdef HAS_LIBGSLIC
        ISuperpixel* superpixel = _superpixel.Compute(frame);
#else
        ISuperpixel* superpixel = _superpixel.Compute(frame);
#endif
        superpixel->GetLabels(superpixel_labels);
        superpixel_id = superpixel_labels.at<unsigned int>(pointer_y, pointer_x);
        superpixel_selected = superpixel_labels == superpixel_id;
        switch (sel.mode) {
            case SuperpixelSelection::Mode::None:
            superpixel->GetContour(superpixel_contour);
            frame_rgb.setTo(cv::Scalar(200, 5, 240), superpixel_contour);
            break;

            case SuperpixelSelection::Mode::Spotlight:
            cv_misc::fx::spotlight(frame_rgb, superpixel_selected, 0.5);
            superpixel->GetContour(superpixel_contour);
            frame_rgb.setTo(cv::Scalar(200, 5, 240), superpixel_contour);
            break;

            case SuperpixelSelection::Mode::Contour:
            std::vector<std::vector<cv::Point>> superpixel_sel_contour;
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
    GSLIC _superpixel;
#else
    OpenCVSLIC _superpixel;
#endif
    VGG16 dcnn;

    bool _is_shown = false;
    char _title[64];
    int pointer_x = 0, pointer_y = 0;
    unsigned int superpixel_id = 0;
    SuperpixelSelection sel = {SuperpixelSelection::Mode::Contour};
    bool use_magnifier = false;
    bool normalize_component = true;

    cv::Mat frame, frame_rgb, frame_dcnn;
    cv::Mat superpixel_contour, superpixel_labels, superpixel_selected;
    cv::Mat histogram_rgb;
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

int main(int, char**) {
    App app = App::Initialize();
    if (!app.ok) return 1;
    ImVec4 clear_color = ImVec4(0.45f, 0.55f, 0.60f, 1.00f);
    cameras = cv_misc::camera_enumerate2();
    windows.push_back(std::make_unique<PipelineSettingsWindow>());
    windows.push_back(std::make_unique<MomentsExperiment>());

    while (app.EventLoop()){
        for (auto w = windows.begin(); w != windows.end();) {
            if (!(*w)->Draw()) windows.erase(w++);
            else ++w;
        }
        app.Render(clear_color);
    }

    // Ensure dtors are orderly invoked: heavy RAII usage
    windows.clear();
    return 0;
}
