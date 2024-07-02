#pragma once
#include <cstdint>
#include <string_view>
#include <functional>
#include <memory>
#include "class_decorations.hpp"

class Engine;

class Application
{
public:
   struct CreateParameters
   {
       std::string_view windowTitle;
       bool isFullscreen;

       std::shared_ptr<Engine> engine;
   };

    Application(const CreateParameters& parameters);

    NON_COPYABLE(Application);
    NON_MOVABLE(Application);

    virtual ~Application();
    virtual void Run() = 0;

protected:
    uint32_t _width, _height;
    std::string_view _windowTitle;
    bool _isFullscreen;
    std::shared_ptr<Engine> _engine;

    bool _quit = false;
    bool _paused = false;
    float _timer = 0.0f;
    uint32_t _frameCounter = 0;
    float _frameTimer = 0.0f;
};
