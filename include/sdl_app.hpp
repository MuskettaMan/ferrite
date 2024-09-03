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
    void SetMouseHidden(bool state) override;

    const InputManager& GetInputManager() const override;

private:
    SDL_Window* _window;
    SDL_Renderer* _renderer;

    InitInfo _initInfo;
    class InputManager _inputManager;

    bool _mouseHidden = false;
};