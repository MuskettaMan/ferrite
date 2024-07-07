#if defined(WINDOWS)
#include "win/win_app.hpp"
#elif defined(LINUX)
#include "linux/linux_app.hpp"
#endif

#include <memory>
#include <iostream>
#include "engine.hpp"
#include "imgui.h"

std::shared_ptr<Application> g_app;
std::shared_ptr<Engine> g_engine;

int main()
{
    g_engine = std::make_shared<Engine>();

    Application::CreateParameters parameters{ "Vulkan", false };

#if defined(WINDOWS)
    g_app = std::make_shared<WinApp>(parameters);
#elif defined(LINUX)
    g_app = std::make_shared<LinuxApp>(parameters);
#endif

    g_engine->Init(g_app->GetInitInfo(), g_app);

    try
    {
        g_app->Run([]() { g_engine->Run(); });
    }
    catch (const std::exception& e)
    {
        std::cerr << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    g_engine->Shutdown();

    return EXIT_SUCCESS;
}

