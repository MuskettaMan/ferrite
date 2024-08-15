#pragma once

#include "include.hpp"

struct Camera
{
    glm::vec3 position;
    glm::quat rotation;
    float fov;

    float nearPlane;
    float farPlane;
};

struct CameraUBO
{
    alignas(16)
    glm::mat4 VP;
    glm::mat4 view;
    glm::mat4 proj;

    alignas(16)
    glm::vec3 cameraPosition;
};

struct CameraStructure
{
    vk::DescriptorSetLayout descriptorSetLayout;
    std::array<vk::DescriptorSet, MAX_FRAMES_IN_FLIGHT> descriptorSets;
    std::array<vk::Buffer, MAX_FRAMES_IN_FLIGHT> buffers;
    std::array<VmaAllocation, MAX_FRAMES_IN_FLIGHT> allocations;
    std::array<void*, MAX_FRAMES_IN_FLIGHT> mappedPtrs;
};
