#pragma once

#include "class_decorations.hpp"
#include "vk_mem_alloc.h"
#include "vulkan_brain.hpp"
#include "vulkan_helper.hpp"
#include <cstdint>
#include <vulkan/vulkan.hpp>
#include <glm/glm.hpp>

template<uint32_t FRAMES, uint32_t N = 4>
class GBuffers
{
public:
    GBuffers(const VulkanBrain& brain, glm::uvec2 size);
    ~GBuffers();

    NON_MOVABLE(GBuffers);
    NON_COPYABLE(GBuffers);

    const std::array<vk::Image, FRAMES>& GBuffersImageArray() const
    {
        return _gBuffersImageArray;
    }

    const std::array<VmaAllocation, FRAMES>& GBufferAllocation() const
    {
        return _gBufferAllocation;
    }

    const std::array<std::array<vk::ImageView, N>, FRAMES>& GBufferViews() const
    {
        return _gBufferViews;
    }

    vk::Image GBuffersImageArray(uint32_t index) const
    {
        return _gBuffersImageArray[index];
    }

    VmaAllocation GBufferAllocation(uint32_t index) const
    {
        return _gBufferAllocation[index];
    }

    const std::array<vk::ImageView, N>& GBufferViews(uint32_t index) const
    {
        return _gBufferViews[index];
    }

    vk::ImageView GBufferView(uint32_t frameIndex, uint32_t viewIndex) const
    {
        return _gBufferViews[frameIndex][viewIndex];
    }

private:
    const VulkanBrain& _brain;
    glm::uvec2 _size;

private:
    std::array<vk::Image, FRAMES> _gBuffersImageArray;
    std::array<VmaAllocation, FRAMES> _gBufferAllocation;
    std::array<std::array<vk::ImageView, N>, FRAMES> _gBufferViews;

    static constexpr std::array<std::string_view, N> _names = {
            "[VIEW] GBuffer RGB: Albedo A: Metallic", "[VIEW] GBuffer RGB: Normal A: Roughness",
            "[VIEW] GBuffer RGB: Emissive A: AO",     "[VIEW] GBuffer RGB: Position A: Unused"
    };

};

template<uint32_t FRAMES, uint32_t N>
GBuffers<FRAMES, N>::GBuffers(const VulkanBrain& brain, glm::uvec2 size) :
    _brain(brain),
    _size(size)
{
    vk::Format format = vk::Format::eR16G16B16A16Sfloat;
    for(size_t i = 0; i < FRAMES; ++i)
    {
        util::CreateImage(_brain.vmaAllocator, size.x, size.y, format,
                          vk::ImageTiling::eOptimal, vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eSampled,
                          _gBuffersImageArray[i], _gBufferAllocation[i], "GBuffer array", N);
        util::NameObject(_gBuffersImageArray[i], "[IMAGE] GBuffer Array", _brain.device, _brain.dldi);

        for(size_t j = 0; j < N; ++j)
        {
            _gBufferViews[i][j] = util::CreateImageView(_brain.device, _gBuffersImageArray[i], format, vk::ImageAspectFlagBits::eColor, j);
            util::NameObject(_gBufferViews[i][j], _names[j], _brain.device, _brain.dldi);
        }

        vk::CommandBuffer cb = util::BeginSingleTimeCommands(_brain.device, _brain.commandPool);
        util::TransitionImageLayout(cb, _gBuffersImageArray[i], format, vk::ImageLayout::eUndefined, vk::ImageLayout::eColorAttachmentOptimal, N);
        util::EndSingleTimeCommands(_brain.device, _brain.graphicsQueue, cb, _brain.commandPool);
    }
}


template<uint32_t FRAMES, uint32_t N>
GBuffers<FRAMES, N>::~GBuffers()
{
    for(size_t i = 0; i < FRAMES; ++i)
    {
        vmaDestroyImage(_brain.vmaAllocator, _gBuffersImageArray[i], _gBufferAllocation[i]);
        for(size_t j = 0; j < N; ++j)
            _brain.device.destroy(_gBufferViews[i][j]);
    }
}
