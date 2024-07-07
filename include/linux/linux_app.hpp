#include "application.hpp"
#include <xcb/xcb.h>
#include <xcb/xproto.h>
#include <X11/Xlib-xcb.h>
#include "engine_init_info.hpp"

class LinuxApp : public Application
{
public:
    LinuxApp(const CreateParameters& parameters);
    ~LinuxApp() override;
    void Run(std::function<void()> updateLoop) override;
    glm::uvec2 DisplaySize() override;
    InitInfo GetInitInfo() override;

private:
    xcb_connection_t* _connection;
    xcb_screen_t* _screen;
    xcb_window_t _window;
    xcb_intern_atom_reply_t* _atomWmDeleteWindow;
    Display* _display;
    Atom _wmDeleteMessage;

    std::vector<const char*> _extensions{ VK_KHR_XCB_SURFACE_EXTENSION_NAME, VK_KHR_SURFACE_EXTENSION_NAME };
};
