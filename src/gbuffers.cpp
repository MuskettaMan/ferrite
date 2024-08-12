#include "gbuffers.hpp"

GBuffers::GBuffers(const VulkanBrain& brain, glm::uvec2 size) :
        _brain(brain),
        _size(size)
{
    auto supportedDepthFormat = util::FindSupportedFormat(_brain.physicalDevice, { vk::Format::eD32Sfloat, vk::Format::eD32SfloatS8Uint, vk::Format::eD24UnormS8Uint },
                                                          vk::ImageTiling::eOptimal,
                                                          vk::FormatFeatureFlagBits::eDepthStencilAttachment);

    assert(supportedDepthFormat.has_value() && "No supported depth format!");

    _depthFormat = supportedDepthFormat.value();

    CreateGBuffers();
    CreateDepthResources();
    CreateViewportAndScissor();
}


GBuffers::~GBuffers()
{
    CleanUp();
}

void GBuffers::Resize(glm::uvec2 size)
{
    if(size == _size)
        return;

    CleanUp();

    _size = size;

    CreateGBuffers();
    CreateDepthResources();
    CreateViewportAndScissor();
}

void GBuffers::CreateGBuffers()
{
    auto format = GBufferFormat();
    for(size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i)
    {
        util::CreateImage(_brain.vmaAllocator, _size.x, _size.y, format,
                          vk::ImageTiling::eOptimal, vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eSampled,
                          _gBuffersImageArray[i], _gBufferAllocation[i], "GBuffer array", DEFERRED_ATTACHMENT_COUNT);
        util::NameObject(_gBuffersImageArray[i], "[IMAGE] GBuffer Array", _brain.device, _brain.dldi);

        for(size_t j = 0; j < DEFERRED_ATTACHMENT_COUNT; ++j)
        {
            _gBufferViews[i][j] = util::CreateImageView(_brain.device, _gBuffersImageArray[i], format, vk::ImageAspectFlagBits::eColor, j);
            util::NameObject(_gBufferViews[i][j], _names[j], _brain.device, _brain.dldi);
        }

        vk::CommandBuffer cb = util::BeginSingleTimeCommands(_brain.device, _brain.commandPool);
        util::TransitionImageLayout(cb, _gBuffersImageArray[i], format, vk::ImageLayout::eUndefined, vk::ImageLayout::eColorAttachmentOptimal, DEFERRED_ATTACHMENT_COUNT);
        util::EndSingleTimeCommands(_brain.device, _brain.graphicsQueue, cb, _brain.commandPool);
    }
}

void GBuffers::CreateDepthResources()
{
    util::CreateImage(_brain.vmaAllocator, _size.x, _size.y,
                      _depthFormat, vk::ImageTiling::eOptimal,
                      vk::ImageUsageFlagBits::eDepthStencilAttachment,
                      _depthImage, _depthImageAllocation, "Depth image");

    _depthImageView = util::CreateImageView(_brain.device, _depthImage, _depthFormat, vk::ImageAspectFlagBits::eDepth);

    vk::CommandBuffer commandBuffer = util::BeginSingleTimeCommands(_brain.device, _brain.commandPool);
    util::TransitionImageLayout(commandBuffer, _depthImage, _depthFormat, vk::ImageLayout::eUndefined, vk::ImageLayout::eDepthStencilAttachmentOptimal);
    util::EndSingleTimeCommands(_brain.device, _brain.graphicsQueue, commandBuffer, _brain.commandPool);
}

void GBuffers::CleanUp()
{
    for(size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i)
    {
        vmaDestroyImage(_brain.vmaAllocator, _gBuffersImageArray[i], _gBufferAllocation[i]);
        for(size_t j = 0; j < DEFERRED_ATTACHMENT_COUNT; ++j)
            _brain.device.destroy(_gBufferViews[i][j]);
    }

    _brain.device.destroy(_depthImageView);
    vmaDestroyImage(_brain.vmaAllocator, _depthImage, _depthImageAllocation);
}

void GBuffers::CreateViewportAndScissor()
{
    _viewport = vk::Viewport{ 0.0f, 0.0f, static_cast<float>(_size.x), static_cast<float>(_size.y), 0.0f,
                              1.0f };
    vk::Extent2D extent{ _size.x, _size.y };

    _scissor = vk::Rect2D{ vk::Offset2D{ 0, 0 }, extent };
}
