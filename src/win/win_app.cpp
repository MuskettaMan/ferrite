#include "win/win_app.hpp"
#include "engine.hpp"
#include "GLFW/glfw3.h"
#include "vulkan_helper.hpp"

WinApp::WinApp(const CreateParameters& parameters) : Application(parameters)
{
    glfwInit();

    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    glfwWindowHint(GLFW_RESIZABLE, !_isFullscreen);
    glfwWindowHint(GLFW_MAXIMIZED, GLFW_TRUE);

    const auto monitor = glfwGetPrimaryMonitor();
    const GLFWvidmode* mode = glfwGetVideoMode(monitor);
    glfwWindowHint(GLFW_RED_BITS, mode->redBits);
    glfwWindowHint(GLFW_GREEN_BITS, mode->greenBits);
    glfwWindowHint(GLFW_BLUE_BITS, mode->blueBits);
    glfwWindowHint(GLFW_REFRESH_RATE, mode->refreshRate);

    if(_isFullscreen)
    {

    }

    _window = glfwCreateWindow(mode->width, mode->height, _windowTitle.data(), nullptr, nullptr);

    _width = mode->width;
    _height = mode->height;

    uint32_t glfwExtensionCount{ 0 };
    const char** glfwExtensions{ glfwGetRequiredInstanceExtensions(&glfwExtensionCount) };

    Engine::InitInfo initInfo{ glfwExtensionCount, glfwExtensions };
    initInfo.retrieveSurface = [this](vk::Instance instance) {
        VkSurfaceKHR surface;
        util::VK_ASSERT(glfwCreateWindowSurface(instance, this->_window, nullptr, &surface), "Failed creating GLFW surface!");
        return vk::SurfaceKHR(surface);
    };

    _engine->Init(initInfo);
}

void WinApp::Run()
{
    while(!glfwWindowShouldClose(_window) || _quit)
    {
        glfwPollEvents();

        _engine->Run();
    }
}

WinApp::~WinApp()
{
    _engine->Shutdown();

    glfwDestroyWindow(_window);
    glfwTerminate();
}
