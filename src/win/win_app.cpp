#include "win/win_app.hpp"
#include "GLFW/glfw3.h"
#include "vulkan_helper.hpp"
#include "imgui/backends/win/imgui_impl_glfw.h"

WinApp::WinApp(const CreateParameters& parameters) : Application(parameters)
{
    glfwInit();

    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    glfwWindowHint(GLFW_RESIZABLE, true);
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


    int32_t width, height;
    glfwGetWindowSize(_window, &width, &height);
    _width = width;
    _height = height;

    double xPos, yPos;
    glfwGetCursorPos(_window, &xPos, &yPos);
    _mousePos.x = xPos;
    _mousePos.y = yPos;
    _lastMousePos = _mousePos;
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

        double xPos, yPos;
        glfwGetCursorPos(_window, &xPos, &yPos);
        _mousePos.x = xPos;
        _mousePos.y = yPos;

        if(glfwGetKey(_window, GLFW_KEY_ESCAPE) && !_escapePressedPreviousFrame)
        {
            _escapePressedPreviousFrame = true;
            _cursorDisabled = false;
            glfwSetInputMode(_window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
        }
        else
            _escapePressedPreviousFrame = false;

        if(glfwGetMouseButton(_window, 0))
        {
            _cursorDisabled = true;
            _lastMousePos = _mousePos;
            glfwSetInputMode(_window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
        }

        updateLoop();

        if(_cursorDisabled)
            _lastMousePos = _mousePos;
    }
}

WinApp::~WinApp()
{
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

    return initInfo;
}

bool WinApp::IsMinimized()
{
    int32_t result = glfwGetWindowAttrib(_window, GLFW_ICONIFIED);
    return static_cast<bool>(result);
}

void WinApp::InitImGui()
{
    ImGui_ImplGlfw_InitForVulkan(_window, true);
}

void WinApp::NewImGuiFrame()
{
    ImGui_ImplGlfw_NewFrame();
}

void WinApp::ShutdownImGui()
{
    ImGui_ImplGlfw_Shutdown();
}

glm::vec2 WinApp::GetMousePosition()
{
    if(!_cursorDisabled)
        return _lastMousePos;

    return _mousePos;
}

glm::vec2 WinApp::GetLastMousePosition()
{
    return _lastMousePos;
}

bool WinApp::KeyPressed(uint32_t keyCode)
{
    return glfwGetKey(_window, keyCode);
}

