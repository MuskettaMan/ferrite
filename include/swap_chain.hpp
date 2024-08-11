#pragma once

#include <vulkan/vulkan.hpp>
#include <glm/glm.hpp>
#include "class_decorations.hpp"
#include "vk_mem_alloc.h"

struct QueueFamilyIndices;

class SwapChain
{
public:
    struct SupportDetails
    {
        vk::SurfaceCapabilitiesKHR capabilities;
        std::vector<vk::SurfaceFormatKHR> formats;
        std::vector<vk::PresentModeKHR> presentModes;
    };

    SwapChain(vk::Device device, vk::PhysicalDevice physicalDevice, vk::Instance instance, vk::CommandPool commandPool, VmaAllocator allocator, vk::Queue graphicsQueue, vk::SurfaceKHR surface, vk::Format depthFormat);
    ~SwapChain();
    NON_MOVABLE(SwapChain);
    NON_COPYABLE(SwapChain);

    void CreateSwapChain(const glm::uvec2& screenSize, const QueueFamilyIndices& familyIndices);
    void RecreateSwapChain(const glm::uvec2& screenSize, const QueueFamilyIndices& familyIndices);

    [[nodiscard]]
    static SupportDetails QuerySupport(vk::PhysicalDevice device, vk::SurfaceKHR surface);

    [[nodiscard]]
    size_t GetImageCount() const { return _images.size(); };

    [[nodiscard]]
    vk::SwapchainKHR GetSwapChain() const { return _swapChain; }

    [[nodiscard]]
    vk::ImageView GetImageView(uint32_t index) const { return _imageViews[index]; }

    [[nodiscard]]
    vk::Extent2D GetExtent() const { return _extent; }

    [[nodiscard]]
    vk::Format GetFormat() const { return _format; }

    [[nodiscard]]
    vk::Image GetImage(uint32_t index) const { return _images[index]; }

    [[nodiscard]]
    vk::Format GetDepthFormat() const { return _depthFormat; }

    [[nodiscard]]
    vk::Image GetDepthImage() const { return _depthImage; }

    [[nodiscard]]
    vk::ImageView GetDepthView() const { return _depthImageView; }

    [[nodiscard]]
    glm::uvec2 GetImageSize() const { return _imageSize; }

private:
    vk::Instance _instance;
    vk::Device _device;
    vk::PhysicalDevice _physicalDevice;
    vk::CommandPool _commandPool;
    vk::Queue _graphicsQueue;
    vk::SurfaceKHR _surface;
    VmaAllocator _allocator;
    glm::uvec2 _imageSize;

    vk::SwapchainKHR _swapChain;
    vk::Extent2D _extent;

    std::vector<vk::Image> _images;
    std::vector<vk::ImageView> _imageViews;
    vk::Format _format;

    vk::Image _depthImage;
    VmaAllocation _depthImageAllocation;
    vk::ImageView _depthImageView;
    vk::Format _depthFormat;

    void CleanUpSwapChain();
    vk::SurfaceFormatKHR ChooseSwapSurfaceFormat(const std::vector<vk::SurfaceFormatKHR>& availableFormats);
    vk::PresentModeKHR ChoosePresentMode(const std::vector<vk::PresentModeKHR>& availablePresentModes);
    vk::Extent2D ChooseSwapExtent(const vk::SurfaceCapabilitiesKHR& capabilities, const glm::uvec2& screenSize);
    void CreateSwapChainImageViews();
    void CreateDepthResources(const glm::uvec2& screenSize);
};