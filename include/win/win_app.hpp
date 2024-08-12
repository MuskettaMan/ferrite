#include "application.hpp"

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

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

private:
    GLFWwindow* _window;
};
