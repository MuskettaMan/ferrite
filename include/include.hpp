#pragma once
#include <iostream>
#include <cstdint>
#include <vector>
#include <algorithm>
#include <cstring>
#include <set>
#include <thread>
#include <spdlog/spdlog.h>
#include <spdlog/fmt/bundled/printf.h>
#include <memory>
#include <optional>
#include <functional>
#include <chrono>
#include <vulkan/vulkan.hpp>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <stb_image.h>
#include "vk_mem_alloc.h"

#include "class_decorations.hpp"
#include "vulkan_helper.hpp"
#include "vulkan_brain.hpp"

constexpr uint32_t MAX_FRAMES_IN_FLIGHT{ 3 };
constexpr uint32_t DEFERRED_ATTACHMENT_COUNT{ 4 };