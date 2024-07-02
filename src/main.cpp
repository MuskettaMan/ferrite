#if defined(WINDOWS)
#include "win/win_app.hpp"
#elif defined(LINUX)
#include "linux/linux_app.hpp"
#endif

#include <memory>
#include <vulkan/vulkan.h>
#include <iostream>
#include "engine.hpp"

std::unique_ptr<Application> g_app;
std::shared_ptr<Engine> g_engine;

int main()
{
    g_engine = std::make_shared<Engine>();

    Application::CreateParameters parameters{ "Vulkan", false, g_engine };

#if defined(WINDOWS)
    g_app = std::make_unique<WinApp>(parameters);
#elif defined(LINUX)
    g_app = std::make_unique<LinuxApp>(parameters);
#endif

    try
    {
        g_app->Run();
    }
    catch (const std::exception& e)
    {
        std::cerr << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}

