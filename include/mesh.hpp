#pragma once

#include <array>
#include <vulkan/vulkan.hpp>
#include <glm/glm.hpp>
#include "vk_mem_alloc.h"

struct Vertex
{
    enum Enumeration {
        ePOSITION,
        eNORMAL,
        eTANGENT,
        eCOLOR,
        eTEX_COORD
    };

    glm::vec3 position;
    glm::vec3 normal;
    glm::vec4 tangent;
    glm::vec3 color;
    glm::vec2 texCoord;

    static vk::VertexInputBindingDescription GetBindingDescription();
    static std::array<vk::VertexInputAttributeDescription, 5> GetAttributeDescriptions();
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

struct Texture
{
    uint32_t width, height, numChannels;
    std::vector<std::byte> data;

    vk::Format GetFormat() const
    {
        return vk::Format::eR8G8B8A8Srgb;
    }
};

struct Material
{
    uint32_t albedoIndex;
    glm::vec3 albedoFactor;
    uint32_t albedoUVChannel;

    uint32_t metallicRoughnessIndex;
    float metallicFactor;
    float roughnessFactor;
    uint32_t metallicRoughnessUVChannel;

    uint32_t normalIndex;
    float normalScale;
    uint32_t normalUVChannel;

    uint32_t occlusionIndex;
    float occlusionStrength;
    uint32_t occlusionUVChannel;

    uint32_t emissiveIndex;
    glm::vec3 emissiveFactor;
    uint32_t emissiveUVChannel;
};

struct Model
{
    std::vector<Mesh> meshes;
    std::vector<Material> materials;
    std::vector<Texture> textures;
};

struct MeshPrimitiveHandle
{
    vk::PrimitiveTopology topology;
    vk::IndexType indexType;
    uint32_t triangleCount;

    vk::Buffer vertexBuffer;
    vk::Buffer indexBuffer;
    VmaAllocation vertexBufferAllocation;
    VmaAllocation indexBufferAllocation;
};

struct MeshHandle
{
    std::vector<MeshPrimitiveHandle> primitives;
};

struct TextureHandle
{
    std::string name;
    vk::Image image;
    VmaAllocation imageAllocation;
    vk::ImageView imageView;
    uint32_t width, height, numChannels;
    vk::Format format;
};

struct ModelHandle
{
    std::vector<MeshHandle> meshes;
    std::vector<Material> materials;
    std::vector<TextureHandle> textures;
};
