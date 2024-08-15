#pragma once
#include "include.hpp"

struct HDRTarget
{
    vk::Format format;
    glm::uvec2 size;
    std::array<vk::Image, MAX_FRAMES_IN_FLIGHT> images;
    std::array<vk::ImageView, MAX_FRAMES_IN_FLIGHT> imageViews;
    std::array<VmaAllocation, MAX_FRAMES_IN_FLIGHT> allocations;
};