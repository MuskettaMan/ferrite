#pragma once

#include "class_decorations.hpp"
#include "vk_mem_alloc.h"
#include "vulkan_brain.hpp"
#include "vulkan_helper.hpp"
#include <cstdint>
#include <vulkan/vulkan.hpp>
#include <glm/glm.hpp>
#include "include.hpp"

class GBuffers
{
public:
    GBuffers(const VulkanBrain& brain, glm::uvec2 size);
    ~GBuffers();

    NON_MOVABLE(GBuffers);
    NON_COPYABLE(GBuffers);

    void Resize(glm::uvec2 size);

    const std::array<vk::Image, MAX_FRAMES_IN_FLIGHT>& GBuffersImageArray() const { return _gBuffersImageArray; }
    const std::array<VmaAllocation, MAX_FRAMES_IN_FLIGHT>& GBufferAllocation() const { return _gBufferAllocation; }
    const std::array<std::array<vk::ImageView, DEFERRED_ATTACHMENT_COUNT>, MAX_FRAMES_IN_FLIGHT>& GBufferViews() const { return _gBufferViews; }
    vk::Image GBuffersImageArray(uint32_t index) const { return _gBuffersImageArray[index]; }
    VmaAllocation GBufferAllocation(uint32_t index) const { return _gBufferAllocation[index]; }
    const std::array<vk::ImageView, DEFERRED_ATTACHMENT_COUNT>& GBufferViews(uint32_t index) const  { return _gBufferViews[index]; }
    vk::ImageView GBufferView(uint32_t frameIndex, uint32_t viewIndex) const { return _gBufferViews[frameIndex][viewIndex]; }
    vk::Format DepthFormat() const { return _depthFormat; }
    glm::uvec2 Size() const { return _size; }
    vk::Image DepthImage() const { return _depthImage; }
    vk::ImageView DepthImageView() const { return _depthImageView; }
    const vk::Rect2D& Scissor() const { return _scissor; }
    const vk::Viewport& Viewport() const { return _viewport; }

    static vk::Format GBufferFormat() { return vk::Format::eR16G16B16A16Sfloat; }

private:
    const VulkanBrain& _brain;
    glm::uvec2 _size;

    std::array<vk::Image, MAX_FRAMES_IN_FLIGHT> _gBuffersImageArray;
    std::array<VmaAllocation, MAX_FRAMES_IN_FLIGHT> _gBufferAllocation;
    std::array<std::array<vk::ImageView, DEFERRED_ATTACHMENT_COUNT>, MAX_FRAMES_IN_FLIGHT> _gBufferViews;

    vk::Image _depthImage;
    VmaAllocation _depthImageAllocation;
    vk::ImageView _depthImageView;
    vk::Format _depthFormat;

    vk::Viewport _viewport;
    vk::Rect2D _scissor;

    static constexpr std::array<std::string_view, DEFERRED_ATTACHMENT_COUNT> _names = {
            "[VIEW] GBuffer RGB: Albedo A: Metallic", "[VIEW] GBuffer RGB: Normal A: Roughness",
            "[VIEW] GBuffer RGB: Emissive A: AO",     "[VIEW] GBuffer RGB: Position A: Unused"
    };

    void CreateGBuffers();
    void CreateDepthResources();
    void CreateViewportAndScissor();
    void CleanUp();
};