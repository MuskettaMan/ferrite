cmake_minimum_required(VERSION 3.25.1)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake")

set(NAME pi_vulkan)

project(pi_vulkan)

set(CMAKE_CXX_STANDARD 23)

#include_directories(external/vulkan)
include_directories(external)
include_directories(external/glm)
add_subdirectory(external/glm)

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


    find_package(xcb REQUIRED)
    if(xcb_FOUND)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DVK_USE_PLATFORM_XCB_KHR")
        message(STATUS "Found xcb")
        include_directories(${XCB_INCLUDE_DIRS})
        link_libraries(${XCB_LIBRARIES})
    endif ()
endif()

if(NOT Vulkan_FOUND)
    message(FATAL_ERROR "Could not find Vulkan library!")
else()
    message(STATUS ${Vulkan_LIBRARY})
endif()

# Link Vulkan to executable
link_libraries(${Vulkan_LIBRARIES})

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

target_link_libraries(pi_vulkan PRIVATE magic_enum::magic_enum)
target_include_directories(pi_vulkan PUBLIC include)
target_compile_options(pi_vulkan PRIVATE -fexceptions -frtti)

