#include "swap_chain.hpp"
#include "vulkan_helper.hpp"
#include "vulkan/vulkan.h"
#include "engine.hpp"

SwapChain::SwapChain(vk::Device device, vk::PhysicalDevice physicalDevice, vk::Instance instance, vk::CommandPool commandPool, vk::Queue graphicsQueue, vk::SurfaceKHR surface, vk::Format depthFormat) :
    _device(device),
    _physicalDevice(physicalDevice),
    _instance(instance),
    _commandPool(commandPool),
    _graphicsQueue(graphicsQueue),
    _surface(surface),
    _depthFormat(depthFormat)
{
}

SwapChain::~SwapChain()
{
    CleanUpSwapChain();
}

void SwapChain::CreateSwapChain(const glm::uvec2& screenSize, const QueueFamilyIndices& familyIndices)
{
    _imageSize = screenSize;
    SupportDetails swapChainSupport = QuerySupport(_physicalDevice, _surface);

    auto surfaceFormat = ChooseSwapSurfaceFormat(swapChainSupport.formats);
    auto presentMode = ChoosePresentMode(swapChainSupport.presentModes);
    auto extent = ChooseSwapExtent(swapChainSupport.capabilities, screenSize);

    uint32_t imageCount = swapChainSupport.capabilities.minImageCount + 1;
    if(swapChainSupport.capabilities.maxImageCount > 0 && imageCount > swapChainSupport.capabilities.maxImageCount)
        imageCount = swapChainSupport.capabilities.maxImageCount;

    vk::SwapchainCreateInfoKHR createInfo{};
    createInfo.surface = _surface;
    createInfo.minImageCount = imageCount + 1;
    createInfo.imageFormat = surfaceFormat.format;
    createInfo.imageColorSpace = surfaceFormat.colorSpace;
    createInfo.imageExtent = extent;
    createInfo.imageArrayLayers = 1;
    createInfo.imageUsage = vk::ImageUsageFlagBits::eColorAttachment; // TODO: Can change this later to a memory transfer operation, when doing post-processing.
    if(swapChainSupport.capabilities.supportedUsageFlags & vk::ImageUsageFlagBits::eTransferSrc)
        createInfo.imageUsage |= vk::ImageUsageFlagBits::eTransferSrc;
    if(swapChainSupport.capabilities.supportedUsageFlags & vk::ImageUsageFlagBits::eTransferDst)
        createInfo.imageUsage |= vk::ImageUsageFlagBits::eTransferDst;

    uint32_t queueFamilyIndices[] = { familyIndices.graphicsFamily.value(), familyIndices.presentFamily.value() };
    if(familyIndices.graphicsFamily != familyIndices.presentFamily)
    {
        createInfo.imageSharingMode = vk::SharingMode::eConcurrent;
        createInfo.queueFamilyIndexCount = 2;
        createInfo.pQueueFamilyIndices = queueFamilyIndices;
    }
    else
    {
        createInfo.imageSharingMode = vk::SharingMode::eExclusive;
        createInfo.queueFamilyIndexCount = 0;
        createInfo.pQueueFamilyIndices = nullptr;
    }

    createInfo.preTransform = swapChainSupport.capabilities.currentTransform;
    createInfo.compositeAlpha = vk::CompositeAlphaFlagBitsKHR::eOpaque;
    createInfo.presentMode = presentMode;
    createInfo.clipped = vk::True;
    createInfo.oldSwapchain = nullptr;

    util::VK_ASSERT(_device.createSwapchainKHR(&createInfo, nullptr, &_swapChain), "Failed creating swap chain!");

    vk::DispatchLoaderDynamic dldi = vk::DispatchLoaderDynamic{ _instance, vkGetInstanceProcAddr, _device, vkGetDeviceProcAddr };
    util::NameObject(_swapChain, "Main Swapchain", _device, dldi);

    _images = _device.getSwapchainImagesKHR(_swapChain);
    _format = surfaceFormat.format;
    _extent = extent;

    CreateSwapChainImageViews();
    CreateDepthResources(screenSize);
}

void SwapChain::RecreateSwapChain(const glm::uvec2& screenSize, const QueueFamilyIndices& familyIndices)
{
    _device.waitIdle();

    CleanUpSwapChain();

    CreateSwapChain(screenSize, familyIndices);
}

void SwapChain::CreateSwapChainImageViews()
{
    _imageViews.resize(_images.size());
    for(size_t i = 0; i < _imageViews.size(); ++i)
    {
        vk::ImageViewCreateInfo createInfo{
                vk::ImageViewCreateFlags{},
                _images[i],
                vk::ImageViewType::e2D,
                _format,
                vk::ComponentMapping{ vk::ComponentSwizzle::eIdentity, vk::ComponentSwizzle::eIdentity, vk::ComponentSwizzle::eIdentity, vk::ComponentSwizzle::eIdentity },
                vk::ImageSubresourceRange{
                        vk::ImageAspectFlagBits::eColor, // aspect mask
                        0, // base mip level
                        1, // level count
                        0, // base array level
                        1  // layer count
                }
        };
        util::VK_ASSERT(_device.createImageView(&createInfo, nullptr, &_imageViews[i]), "Failed creating image view for swap chain!");

        vk::DispatchLoaderDynamic dldi = vk::DispatchLoaderDynamic{ _instance, vkGetInstanceProcAddr, _device, vkGetDeviceProcAddr };
        util::NameObject(_imageViews[i], "Swapchain Image View", _device, dldi);
        util::NameObject(_images[i], "Swapchain Image", _device, dldi);
    }
}

SwapChain::SupportDetails SwapChain::QuerySupport(vk::PhysicalDevice device, vk::SurfaceKHR surface)
{
    SupportDetails details{};

    util::VK_ASSERT(device.getSurfaceCapabilitiesKHR(surface, &details.capabilities), "Failed getting surface capabilities from physical device!");

    details.formats = device.getSurfaceFormatsKHR(surface);
    details.presentModes = device.getSurfacePresentModesKHR(surface);

    return details;
}

vk::SurfaceFormatKHR SwapChain::ChooseSwapSurfaceFormat(const std::vector<vk::SurfaceFormatKHR>& availableFormats)
{
    for(const auto& format : availableFormats)
    {
        if(format.format == vk::Format::eB8G8R8A8Unorm && format.colorSpace == vk::ColorSpaceKHR::eSrgbNonlinear)
            return format;
    }

    return availableFormats[0];
}

vk::PresentModeKHR SwapChain::ChoosePresentMode(const std::vector<vk::PresentModeKHR>& availablePresentModes)
{
    auto it = std::find_if(availablePresentModes.begin(), availablePresentModes.end(),
                           [](const auto& mode) { return mode == vk::PresentModeKHR::eMailbox; });
    if(it != availablePresentModes.end())
        return *it;

    it = std::find_if(availablePresentModes.begin(), availablePresentModes.end(),
                      [](const auto& mode) { return mode == vk::PresentModeKHR::eFifo; });
    if(it != availablePresentModes.end())
        return *it;

    return availablePresentModes[0];
}

vk::Extent2D SwapChain::ChooseSwapExtent(const vk::SurfaceCapabilitiesKHR& capabilities, const glm::uvec2& screenSize)
{
    if(capabilities.currentExtent.width != std::numeric_limits<uint32_t>::max())
        return capabilities.currentExtent;

    vk::Extent2D extent = { screenSize.x, screenSize.y };
    extent.width = std::clamp(extent.width, capabilities.minImageExtent.width, capabilities.maxImageExtent.width);
    extent.height = std::clamp(extent.height, capabilities.minImageExtent.height, capabilities.maxImageExtent.height);

    return extent;
}

void SwapChain::CleanUpSwapChain()
{
    for(auto imageView : _imageViews)
        _device.destroy(imageView);

    _device.destroy(_swapChain);

    _device.destroy(_depthImage);
    _device.destroy(_depthImageView);
    _device.freeMemory(_depthImageMemory);
}

void SwapChain::CreateDepthResources(const glm::uvec2& screenSize)
{
    util::CreateImage(_device, _physicalDevice,
                      screenSize.x, screenSize.y,
                      _depthFormat, vk::ImageTiling::eOptimal,
                      vk::ImageUsageFlagBits::eDepthStencilAttachment,
                      vk::MemoryPropertyFlagBits::eDeviceLocal,
                      _depthImage, _depthImageMemory);

    _depthImageView = util::CreateImageView(_device, _depthImage, _depthFormat, vk::ImageAspectFlagBits::eDepth);

    vk::CommandBuffer commandBuffer = util::BeginSingleTimeCommands(_device, _commandPool);
    util::TransitionImageLayout(commandBuffer, _depthImage, _depthFormat, vk::ImageLayout::eUndefined, vk::ImageLayout::eDepthStencilAttachmentOptimal);
    util::EndSingleTimeCommands(_device, _graphicsQueue, commandBuffer, _commandPool);
}

