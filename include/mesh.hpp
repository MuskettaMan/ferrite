#pragma once

#include <array>
#include <vulkan/vulkan.hpp>
#include <glm/glm.hpp>

struct Vertex
{
    enum Enumeration {
        ePOSITION,
        eCOLOR,
        eTEX_COORD
    };

    glm::vec3 position;
    glm::vec3 color;
    glm::vec2 texCoord;

    static vk::VertexInputBindingDescription GetBindingDescription();
    static std::array<vk::VertexInputAttributeDescription, 3> GetAttributeDescriptions();
};

struct MeshPrimitive
{
    vk::PrimitiveTopology topology;

    vk::IndexType indexType;
    std::vector<std::byte> indices;
    std::vector<Vertex> vertices;
};

struct Mesh
{
    std::vector<MeshPrimitive> primitives;
};

using Model = std::vector<Mesh>;

struct MeshPrimitiveHandle
{
    vk::PrimitiveTopology topology;
    vk::IndexType indexType;
    uint32_t triangleCount;

    vk::Buffer vertexBuffer;
    vk::Buffer indexBuffer;
    vk::DeviceMemory vertexBufferMemory;
    vk::DeviceMemory indexBufferMemory;
};

struct MeshHandle
{
    std::vector<MeshPrimitiveHandle> primitives;
};

using ModelHandle = std::vector<MeshHandle>;
