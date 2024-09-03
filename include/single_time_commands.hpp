#pragma once
#include "class_decorations.hpp"
#include "vulkan/vulkan.hpp"
#include "vk_mem_alloc.h"

class VulkanBrain;
struct Texture;
struct TextureHandle;

class SingleTimeCommands
{
public:
    SingleTimeCommands(const VulkanBrain& brain);
    ~SingleTimeCommands();

    void Submit();
    void CreateTextureImage(const Texture& texture, TextureHandle& textureHandle, bool generateMips);
    void CreateLocalBuffer(const std::byte* vec, uint32_t count, vk::Buffer& buffer, VmaAllocation& allocation, vk::BufferUsageFlags usage, std::string_view name);

    template <typename T>
    void CreateLocalBuffer(const std::vector<T>& vec, vk::Buffer& buffer, VmaAllocation& allocation, vk::BufferUsageFlags usage, std::string_view name)
    {
        CreateLocalBuffer(reinterpret_cast<const std::byte*>(vec.data()), vec.size() * sizeof(T), buffer, allocation, usage, name);
    }

    const vk::CommandBuffer& CommandBuffer() const { return _commandBuffer; }

    NON_MOVABLE(SingleTimeCommands);
    NON_COPYABLE(SingleTimeCommands);

private:
    const VulkanBrain& _brain;
    vk::CommandBuffer _commandBuffer;
    bool _submitted{ false };

    std::vector<vk::Buffer> _stagingBuffers;
    std::vector<VmaAllocation> _stagingAllocations;
};