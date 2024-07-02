#include "application.hpp"
#include <xcb/xcb.h>
#include <xcb/xproto.h>

class LinuxApp : public Application
{
public:
    LinuxApp(const CreateParameters& parameters);
    ~LinuxApp() override;
    void Run() override;

private:
    xcb_connection_t* _connection;
    xcb_screen_t* _screen;
    xcb_window_t _window;
    xcb_intern_atom_reply_t* _atomWmDeleteWindow;
};
