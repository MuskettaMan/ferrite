#include "application.hpp"

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>

class WinApp : public Application
{
public:
    WinApp(const CreateParameters& parameters);
    ~WinApp() override;
    void Run(std::function<void()> updateLoop) override;
    glm::uvec2 DisplaySize() override;
    InitInfo GetInitInfo() override;
    bool IsMinimized() override;
    void InitImGui() override;
    void NewImGuiFrame() override;
    void ShutdownImGui() override;
    glm::vec2 GetMousePosition() override;
    glm::vec2 GetLastMousePosition() override;
    bool KeyPressed(uint32_t keyCode) override;

private:
    GLFWwindow* _window;

    glm::vec2 _mousePos;
    glm::vec2 _lastMousePos;
    bool _cursorDisabled{ false };
    bool _escapePressedPreviousFrame;
};
