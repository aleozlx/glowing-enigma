#ifndef __WM_HPP__
#define __WM_HPP__
#include <string>
#include "imgui.h"

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

class IStaticWindow: public IWindow {
    public:
    virtual IWindow* Show() override {
        return dynamic_cast<IWindow*>(this);
    }
    virtual ~IStaticWindow() {}
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

#endif
