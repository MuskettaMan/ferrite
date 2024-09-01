#pragma once
#include "application.hpp"
#include "class_decorations.hpp"
#include "SDL3/SDL.h"

class SDLApp : public Application
{
public:
    SDLApp(const CreateParameters& parameters);
    ~SDLApp() override;

    NON_COPYABLE(SDLApp);
    NON_MOVABLE(SDLApp);

    InitInfo GetInitInfo() override;

    glm::uvec2 DisplaySize() override;

    bool IsMinimized() override;

    void Run(std::function<void()> updateLoop) override;

    void InitImGui() override;

    void NewImGuiFrame() override;

    void ShutdownImGui() override;

    glm::vec2 GetMousePosition() override;

    glm::vec2 GetLastMousePosition() override;

    bool KeyPressed(uint32_t keyCode) override;

private:
    SDL_Window* _window;
    SDL_Renderer* _renderer;

    InitInfo _initInfo;

    glm::vec2 _mousePos;
    glm::vec2 _lastMousePos;
};