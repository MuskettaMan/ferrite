#include "linux/linux_app.hpp"

#include <chrono>
#include <cstring>
#include <cassert>
#include <vulkan/vulkan_core.h>
#include <vulkan/vulkan_xcb.h>

#include "vulkan_helper.hpp"
#include "imgui/backends/linux/imgui_impl_xlib.h"

static inline xcb_intern_atom_reply_t* intern_atom_helper(xcb_connection_t* conn, bool only_if_exists, const char* str)
{
    xcb_intern_atom_cookie_t cookie = xcb_intern_atom(conn, only_if_exists, strlen(str), str);
    xcb_flush(conn);
    return xcb_intern_atom_reply(conn, cookie, nullptr);
}

LinuxApp::LinuxApp(const CreateParameters& parameters) : Application(parameters)
{
    _display = XOpenDisplay(nullptr);

    // xcb_connect always returns a non-NULL pointer to a xcb_connection_t,
    // even on failure. Callers need to use xcb_connection_has_error() to
    // check for failure. When finished, use xcb_disconnect() to close the
    // g_connection and free the structure.
    _connection = XGetXCBConnection(_display);
    assert(_connection);

    const xcb_setup_t* setup = xcb_get_setup(_connection);
    xcb_screen_iterator_t iter = xcb_setup_roots_iterator(setup);
    _screen = iter.data;

    uint32_t valueMask, valueList[32];
    _window = xcb_generate_id(_connection);

    valueMask = XCB_CW_BACK_PIXEL | XCB_CW_EVENT_MASK;
    valueList[0] = _screen->black_pixel;
    valueList[1] =
            XCB_EVENT_MASK_KEY_RELEASE |
            XCB_EVENT_MASK_KEY_PRESS |
            XCB_EVENT_MASK_EXPOSURE |
            XCB_EVENT_MASK_STRUCTURE_NOTIFY |
            XCB_EVENT_MASK_POINTER_MOTION |
            XCB_EVENT_MASK_BUTTON_PRESS |
            XCB_EVENT_MASK_BUTTON_RELEASE;

    _width = _screen->width_in_pixels;
    _height = _screen->height_in_pixels;

    xcb_create_window(_connection,
                      XCB_COPY_FROM_PARENT,
                      _window,
                      _screen->root,
                      0, 0,
                      _width, _height,
                      0,
                      XCB_WINDOW_CLASS_INPUT_OUTPUT,
                      _screen->root_visual,
                      valueMask, valueList);

    xcb_intern_atom_reply_t* reply = intern_atom_helper(_connection, true, "WM_PROTOCOLS");
    _atomWmDeleteWindow = intern_atom_helper(_connection, false, "WM_DELETE_WINDOW");

    xcb_change_property(_connection, XCB_PROP_MODE_REPLACE, _window, reply->atom, 4, 32, 1, &_atomWmDeleteWindow->atom);

    xcb_change_property(_connection, XCB_PROP_MODE_REPLACE, _window, XCB_ATOM_WM_NAME, XCB_ATOM_STRING, 8, _windowTitle.size(), _windowTitle.data());

    free(reply);

    std::string wmClass;
    wmClass = wmClass.insert(0, _windowTitle);
    wmClass = wmClass.insert(wmClass.size(), 1, '\0');
    xcb_change_property(_connection, XCB_PROP_MODE_REPLACE, _window, XCB_ATOM_WM_CLASS, XCB_ATOM_STRING, 8, wmClass.size() + 2, wmClass.c_str());

    if(_isFullscreen)
    {
        xcb_intern_atom_reply_t* atomWmState = intern_atom_helper(_connection, false, "_NET_WM_STATE");
        xcb_intern_atom_reply_t* atomWmFulscreen = intern_atom_helper(_connection, false, "_NET_WM_STATE_FULLSCREEN");
        xcb_change_property(_connection, XCB_PROP_MODE_REPLACE, _window, atomWmState->atom, XCB_ATOM_ATOM, 32, 1, &atomWmFulscreen->atom);
        free(atomWmFulscreen);
        free(atomWmState);
    }

    _wmDeleteMessage = XInternAtom(_display, "WM_DELETE_WINDOW", False);
    XSetWMProtocols(_display, _window, &_wmDeleteMessage, 1);

    xcb_map_window(_connection, _window);
    xcb_flush(_connection);

    ImGui_ImplXlib_Init(_display, _window);
}

void LinuxApp::Run(std::function<void()> updateLoop)
{
    while(!_quit)
    {
        while(XPending(_display))
        {
            XEvent event;
            XNextEvent(_display, &event);
            ImGui_ImplXlib_ProcessEvent(&event);

            if(event.type == ClientMessage && event.xclient.window == _window && static_cast<Atom>(event.xclient.data.l[0]) == _wmDeleteMessage)
                _quit = true;
        }

        _width = _screen->width_in_pixels;
        _height = _screen->height_in_pixels;

        updateLoop();
    }
}

LinuxApp::~LinuxApp()
{
    ImGui_ImplXlib_Shutdown();

    xcb_destroy_window(_connection, _window);
    xcb_disconnect(_connection);
}

glm::uvec2 LinuxApp::DisplaySize()
{
    return glm::uvec2(_width, _height);
}

InitInfo LinuxApp::GetInitInfo()
{
    InitInfo initInfo{};
    initInfo.extensions = _extensions.data();
    initInfo.extensionCount = _extensions.size();

    initInfo.width = _width;
    initInfo.height = _height;

    initInfo.retrieveSurface = [this](vk::Instance instance) {
        VkXcbSurfaceCreateInfoKHR createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_XCB_SURFACE_CREATE_INFO_KHR;
        createInfo.connection = this->_connection;
        createInfo.window = this->_window;
        VkSurfaceKHR surface;
        util::VK_ASSERT(vkCreateXcbSurfaceKHR(static_cast<VkInstance>(instance), &createInfo, nullptr, &surface), "Failed creating XCB surface!");

        return vk::SurfaceKHR{ surface };
    };
    initInfo.newImGuiFrame = [](){ ImGui_ImplXlib_NewFrame(); };

    return initInfo;
}
