#pragma once

#include <array>
#include <vulkan/vulkan.hpp>
#include <glm/glm.hpp>
#include "vk_mem_alloc.h"
#include <memory>

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

    uint32_t materialIndex;
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
    glm::vec4 albedoFactor;
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

struct TextureHandle
{
    std::string name;
    vk::Image image;
    VmaAllocation imageAllocation;
    vk::ImageView imageView;
    uint32_t width, height, numChannels;
    vk::Format format;
};

struct MaterialHandle
{
    vk::DescriptorSet descriptorSet;
    std::array<std::shared_ptr<TextureHandle>, 5> textures;

    static std::array<vk::DescriptorSetLayoutBinding, 6> GetLayoutBindings()
    {
        std::array<vk::DescriptorSetLayoutBinding, 6> bindings{};
        bindings[0].binding = 0;
        bindings[0].descriptorType = vk::DescriptorType::eSampler;
        bindings[0].descriptorCount = 1;
        bindings[0].stageFlags = vk::ShaderStageFlagBits::eFragment;
        for(size_t i = 1; i < bindings.size(); ++i)
        {
            bindings[i].binding = i;
            bindings[i].descriptorType = vk::DescriptorType::eSampledImage;
            bindings[i].descriptorCount = 1;
            bindings[i].stageFlags = vk::ShaderStageFlagBits::eFragment;
        }

        return bindings;
    }
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

    std::shared_ptr<MaterialHandle> material;
};

struct MeshHandle
{
    std::vector<MeshPrimitiveHandle> primitives;
};

struct ModelHandle
{
    std::vector<MeshHandle> meshes;
    std::vector<std::shared_ptr<MaterialHandle>> materials;
    std::vector<std::shared_ptr<TextureHandle>> textures;
};
