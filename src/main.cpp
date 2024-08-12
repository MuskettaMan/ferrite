#if defined(WINDOWS)
#include "win/win_app.hpp"
#elif defined(LINUX)
#include "linux/linux_app.hpp"
#endif

#include <memory>
#include <iostream>
#include "engine.hpp"
#include "imgui.h"
#include <spdlog/spdlog.h>

std::shared_ptr<Application> g_app;
std::unique_ptr<Engine> g_engine;

int main()
{

    Application::CreateParameters parameters{ "Vulkan", true };

#if defined(WINDOWS)
    g_app = std::make_shared<WinApp>(parameters);
#elif defined(LINUX)
    g_app = std::make_shared<LinuxApp>(parameters);
#endif

    g_engine = std::make_unique<Engine>(g_app->GetInitInfo(), g_app);

    try
    {
        g_app->Run([]() { g_engine->Run(); });
    }
    catch (const std::exception& e)
    {
        spdlog::error(e.what());
        return EXIT_FAILURE;
    }

    g_engine.reset();

    return EXIT_SUCCESS;
}

