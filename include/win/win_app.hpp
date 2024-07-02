#include "application.hpp"

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

class WinApp : public Application
{
public:
    WinApp(const CreateParameters& parameters);
    ~WinApp() override;
    void Run() override;

private:
    GLFWwindow* _window;
};
