#pragma once
#include <cstdint>
#include <string_view>
#include <functional>
#include <memory>
#include <glm/glm.hpp>
#include "class_decorations.hpp"
#include "engine_init_info.hpp"

class Application
{
public:
   struct CreateParameters
   {
       std::string_view windowTitle;
       bool isFullscreen;
   };

    Application(const CreateParameters& parameters);
    virtual InitInfo GetInitInfo() = 0;
    virtual glm::uvec2 DisplaySize() = 0;
    virtual bool IsMinimized() = 0;

    NON_COPYABLE(Application);
    NON_MOVABLE(Application);

    virtual ~Application();
    virtual void Run(std::function<void()> updateLoop) = 0;
    virtual void InitImGui() = 0;
    virtual void NewImGuiFrame() = 0;
    virtual void ShutdownImGui() = 0;

protected:
    uint32_t _width, _height;
    std::string_view _windowTitle;
    bool _isFullscreen;

    bool _quit = false;
    bool _paused = false;
    float _timer = 0.0f;
    uint32_t _frameCounter = 0;
    float _frameTimer = 0.0f;
};
