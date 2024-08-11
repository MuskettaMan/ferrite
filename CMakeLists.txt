cmake_minimum_required(VERSION 3.25.1)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake")

set(NAME pi_vulkan)

# Static linking only on windows.
if(WIN32)
    set(CMAKE_EXE_LINKER_FLAGS "-static-libstdc++ -static-libgcc -static")
endif()

project(pi_vulkan)

set(CMAKE_CXX_STANDARD 20)

#include_directories(external/vulkan)
include_directories(external)

add_subdirectory(external/magic_enum)

find_package(Vulkan)

if(WIN32)
    add_definitions(-DWINDOWS)
    if(NOT Vulkan_FOUND)
        find_library(Vulkan_LIBRARY NAMES vulkan-1 vulkan PATHS ${CMAKE_SOURCE_DIR}/libs/vulkan)
        if(Vulkan_LIBRARY)

            set(Vulkan_FOUND ON)
            message("WIN32: Using bundled Vulkan library version")
        endif()
    endif()
    set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} -DVK_USE_PLATFORM_WIN32_KHR)
elseif(LINUX)
    add_definitions(-DLINUX)
    if(NOT Vulkan_FOUND)
        find_library(Vulkan_LIBRARY NAMES vulkan HINTS "$ENV{VULKAN_SDK}/lib" "${CMAKE_SOURCE_DIR}/libs/vulkan" REQUIRED)
        if(Vulkan_LIBRARY)
            set(Vulkan_FOUND ON)
            message("LINUX: Using bundled Vulkan library version")
        endif()
    else()
        include_directories(${Vulkan_INCLUDE_DIRS})
    endif()

    find_package(X11 REQUIRED)
    if(X11_FOUND)
        message(STATUS "Found x11")
        include_directories(${X11_INCLUDE_DIRS})
        #link_libraries(${X11_LIBRARIES})
    endif()
    if(NOT X11_xcb_FOUND)
        message(FATAL_ERROR "X11-XCB not found")
    endif()

    find_package(xcb REQUIRED)
    if(xcb_FOUND)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DVK_USE_PLATFORM_XCB_KHR")
        message(STATUS "Found xcb")
        include_directories(${XCB_INCLUDE_DIRS})
        #link_libraries(${XCB_LIBRARIES})
    endif ()

endif()

if(NOT Vulkan_FOUND)
    message(FATAL_ERROR "Could not find Vulkan library!")
else()
    message(STATUS ${Vulkan_LIBRARY})
endif()

# Link Vulkan to executable
link_libraries(${Vulkan_LIBRARY})

# Add and link glfw on Windows
if(WIN32)
    add_subdirectory(external/glfw)
    link_libraries(glfw)
endif()

file(GLOB_RECURSE SOURCE_FILES ${PROJECT_SOURCE_DIR}/src/*.cpp)
file(GLOB_RECURSE SOURCE_FILES_WIN32 ${PROJECT_SOURCE_DIR}/src/win/*.cpp)
file(GLOB_RECURSE SOURCE_FILES_LINUX ${PROJECT_SOURCE_DIR}/src/linux/*.cpp)

# Append platform-specific source files
if(WIN32)
    list(REMOVE_ITEM SOURCE_FILES ${SOURCE_FILES_LINUX})
elseif(LINUX)
    list(REMOVE_ITEM SOURCE_FILES ${SOURCE_FILES_WIN32})
endif()

message(STATUS ${SOURCE_FILES_WIN32})
message(STATUS ${SOURCE_FILES_LINUX})
message(STATUS ${SOURCE_FILES})

list(REMOVE_DUPLICATES SOURCE_FILES)

# Create executable
add_executable(pi_vulkan ${SOURCE_FILES})

target_include_directories(pi_vulkan SYSTEM PRIVATE external/glm)
add_subdirectory(external/glm SYSTEM)

target_link_libraries(pi_vulkan PRIVATE magic_enum::magic_enum)
target_include_directories(pi_vulkan PUBLIC include)
target_compile_options(pi_vulkan PRIVATE -fexceptions -frtti $<$<CONFIG:Debug>:-O0 -g> $<$<CONFIG:Release>:-O3>)
target_link_options(pi_vulkan PRIVATE
        $<$<CONFIG:Release>:-s>
)

# X11 XCB
if(LINUX)
    find_library(XINPUT_LIBRARY NAMES Xi)
    find_path(XINPUT_INCLUDE_DIR NAMES X11/extensions/XInput2.h)

    if(NOT XINPUT_LIBRARY OR NOT XINPUT_INCLUDE_DIR)
        message(FATAL_ERROR "XInput2 not found")
    endif()

    target_link_libraries(pi_vulkan PRIVATE ${X11_LIBRARIES} ${XCB_LIBRARIES} ${X11_xcb_LIB} ${X11_X11_xcb_LIB} ${XINPUT_LIBRARY})
endif()

# END X11 XCB

# FREE TYPE

set(FREETYPE_DIR external/freetype/freetype2)

add_subdirectory(${FREETYPE_DIR})

target_link_libraries(pi_vulkan PUBLIC ${FREETYPE_LIBRARY})
target_include_directories(pi_vulkan SYSTEM PUBLIC ${FREETYPE_DIR}/include)

# END FREETYPE

# IMGUI

set(IMGUI_DIR external/imgui)

set(IMGUI_SOURCE_FILES
        ${IMGUI_DIR}/imgui.cpp
        ${IMGUI_DIR}/imgui_demo.cpp
        ${IMGUI_DIR}/imgui_draw.cpp
        ${IMGUI_DIR}/imgui_tables.cpp
        ${IMGUI_DIR}/imgui_widgets.cpp
        ${IMGUI_DIR}/implot.cpp
        ${IMGUI_DIR}/implot_items.cpp
        ${IMGUI_DIR}/misc/cpp/imgui_stdlib.cpp
        ${IMGUI_DIR}/misc/freetype/imgui_freetype.cpp
        ${IMGUI_DIR}/backends/imgui_impl_vulkan.cpp
)
if(WIN32)
    list(APPEND IMGUI_SOURCE_FILES ${IMGUI_DIR}/backends/win/imgui_impl_glfw.cpp)
elseif(LINUX)
    list(APPEND IMGUI_SOURCE_FILES ${IMGUI_DIR}/backends/linux/imgui_impl_xlib.cpp)
endif()

add_library(imgui STATIC ${IMGUI_SOURCE_FILES})

target_include_directories(imgui PUBLIC ${IMGUI_DIR} ${IMGUI_DIR}/backends)
target_include_directories(imgui PUBLIC ${FREETYPE_DIR}/include)

target_link_libraries(pi_vulkan PRIVATE imgui)
target_link_libraries(pi_vulkan PRIVATE ${XINPUT_LIBRARY})
target_include_directories(pi_vulkan PRIVATE ${IMGUI_DIR} ${IMGUI_DIR}/backends)

# END IMGUI


include_directories("external/stb")

add_subdirectory(external/spdlog)
target_link_libraries(pi_vulkan PRIVATE spdlog::spdlog)

add_subdirectory(external/fastgltf)
target_link_libraries(pi_vulkan PRIVATE fastgltf::fastgltf)


# BUILD TYPE SETTINGS

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

if(CMAKE_BUILD_TYPE MATCHES Debug)
    message("Debug build.")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0")
elseif(CMAKE_BUILD_TYPE MATCHES Release)
    message("Release build.")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")
endif()

if(CMAKE_BUILD_TYPE MATCHES Debug)
    target_compile_definitions(pi_vulkan PRIVATE DEBUG_MODE)
elseif(CMAKE_BUILD_TYPE MATCHES Release)
    target_compile_definitions(pi_vulkan PRIVATE NDEBUG)
endif()

# END BUILD TYPE SETTINGS
