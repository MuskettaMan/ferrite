#include "win/sdl_app.hpp"
#include "SDL3/SDL.h"
#include "SDL3/SDL_vulkan.h"
#include "imgui/backends/imgui_impl_sdl3.h"
#include "imgui/backends/imgui_impl_sdlrenderer3.h"
#include "include.hpp"

SDLApp::SDLApp(const CreateParameters& parameters) : Application(parameters)
{
    if(SDL_Init(SDL_INIT_VIDEO))
    {
        spdlog::error("Failed initializing SDL!");
        return;
    }

    const SDL_DisplayMode* dm = SDL_GetCurrentDisplayMode(0);

    SDL_WindowFlags flags = SDL_WINDOW_VULKAN;
    if(parameters.isFullscreen)
        flags |= SDL_WINDOW_FULLSCREEN;

    _window = SDL_CreateWindow(parameters.windowTitle.data(), dm->w, dm->h, flags);

    if(_window == nullptr)
    {
        spdlog::error("Failed creating SDL window: {}", SDL_GetError());
        SDL_Quit();
        return;
    }

    _renderer = SDL_CreateRenderer(_window, "Vulkan renderer");
    if(_window == nullptr)
    {
        spdlog::error("Failed creating SDL renderer: {}", SDL_GetError());
        SDL_DestroyWindow(_window);
        SDL_Quit();
        return;
    }

    uint32_t sdlExtensionsCount = 0;
    _initInfo.extensions = SDL_Vulkan_GetInstanceExtensions(&sdlExtensionsCount);
    _initInfo.extensionCount = sdlExtensionsCount;

    _initInfo.width = dm->w;
    _initInfo.height = dm->h;
    _initInfo.retrieveSurface = [this](vk::Instance instance) {
        vk::SurfaceKHR surface;
        if(!SDL_Vulkan_CreateSurface(_window, instance, nullptr, reinterpret_cast<VkSurfaceKHR*>(&surface)))
        {
            spdlog::error("Failed creating SDL vk::Surface: {}", SDL_GetError());
        }
        return surface;
    };

    float xPos, yPos;
    SDL_GetMouseState(&xPos, &yPos);
    _mousePos.x = xPos;
    _mousePos.y = yPos;
    _lastMousePos = _mousePos;
}

SDLApp::~SDLApp()
{
    SDL_DestroyWindow(_window);
    SDL_DestroyRenderer(_renderer);
    SDL_Quit();
}

InitInfo SDLApp::GetInitInfo()
{
    return _initInfo;
}

glm::uvec2 SDLApp::DisplaySize()
{
    int32_t w, h;
    SDL_GetWindowSize(_window, &w, &h);
    return glm::uvec2{ w, h };
}

bool SDLApp::IsMinimized()
{
    SDL_WindowFlags flags = SDL_GetWindowFlags(_window);
    return flags & SDL_WINDOW_MINIMIZED;
}

void SDLApp::Run(std::function<void()> updateLoop)
{
    bool running = true;
    while(running)
    {
        SDL_Event event;
        while(SDL_PollEvent(&event))
        {
            if(event.type == SDL_EventType::SDL_EVENT_QUIT)
                running = false;
        }

        float xPos, yPos;
        SDL_GetMouseState(&xPos, &yPos);
        _mousePos.x = xPos;
        _mousePos.y = yPos;

        updateLoop();

        _lastMousePos = _mousePos;
    }
}

void SDLApp::InitImGui()
{
    ImGui_ImplSDL3_InitForVulkan(_window);
    ImGui_ImplSDLRenderer3_Init(_renderer);
}

void SDLApp::NewImGuiFrame()
{
    ImGui_ImplSDL3_NewFrame();
    ImGui_ImplSDLRenderer3_NewFrame();
}

void SDLApp::ShutdownImGui()
{
    ImGui_ImplSDL3_Shutdown();
    ImGui_ImplSDLRenderer3_Shutdown();
}

glm::vec2 SDLApp::GetMousePosition()
{
    return _mousePos;
}

glm::vec2 SDLApp::GetLastMousePosition()
{
    return _lastMousePos;
}

bool SDLApp::KeyPressed(uint32_t keyCode)
{
    const uint8_t* state = SDL_GetKeyboardState(nullptr);

    return state[keyCode];
}
