#include "win/win_app.hpp"
#include "GLFW/glfw3.h"
#include "vulkan_helper.hpp"
#include "imgui/backends/win/imgui_impl_glfw.h"

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

    ImGui_ImplGlfw_InitForVulkan(_window, true);
}

void WinApp::Run(std::function<void()> updateLoop)
{
    while(!glfwWindowShouldClose(_window) || _quit)
    {
        glfwPollEvents();

        int32_t width, height;
        glfwGetWindowSize(_window, &width, &height);

        _width = width;
        _height = height;

        updateLoop();
    }
}

WinApp::~WinApp()
{
    ImGui_ImplGlfw_Shutdown();

    glfwDestroyWindow(_window);
    glfwTerminate();
}

glm::uvec2 WinApp::DisplaySize()
{
    return glm::uvec2(_width, _height);
}

InitInfo WinApp::GetInitInfo()
{
    uint32_t glfwExtensionCount{ 0 };
    const char** glfwExtensions{ glfwGetRequiredInstanceExtensions(&glfwExtensionCount) };

    InitInfo initInfo{ glfwExtensionCount, glfwExtensions };
    initInfo.width = _width;
    initInfo.height = _height;
    initInfo.retrieveSurface = [this](vk::Instance instance) {
        VkSurfaceKHR surface;
        util::VK_ASSERT(glfwCreateWindowSurface(instance, this->_window, nullptr, &surface), "Failed creating GLFW surface!");
        return vk::SurfaceKHR(surface);
    };
    initInfo.newImGuiFrame = [](){ ImGui_ImplGlfw_NewFrame(); };

    return initInfo;
}

