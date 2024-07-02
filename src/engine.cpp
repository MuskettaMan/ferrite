#include <iostream>
#include <cstdint>
#include <cassert>
#include <vector>
#include <algorithm>
#include <cstring>
#include <map>
#include <set>

#include "engine.hpp"
#include "vulkan_validation.hpp"
#include "vulkan_helper.hpp"

Engine::Engine()
{

}

void Engine::Init(const InitInfo& initInfo)
{
    CreateInstance(initInfo);
    SetupDebugMessenger();

    _surface = initInfo.retrieveSurface(_instance);

    PickPhysicalDevice();
    CreateDevice();
    CreateSwapChain(initInfo);
    CreateSwapChainImageViews();
}

void Engine::Run()
{

}

void Engine::Shutdown()
{
    for(auto imageView : _swapChainImageViews)
        _device.destroy(imageView);

    if(_enableValidationLayers)
        util::DestroyDebugUtilsMessengerEXT(_instance, _debugMessenger, nullptr);

    _device.destroy(_swapChain);
    _instance.destroy(_surface);
    _device.destroy();
    _instance.destroy();
}

void Engine::CreateInstance(const InitInfo& initInfo)
{
    CheckValidationLayerSupport();
    if(_enableValidationLayers && !CheckValidationLayerSupport())
        throw std::runtime_error("Validation layers requested, but not supported!");

    vk::ApplicationInfo appInfo{ "", vk::makeApiVersion(0, 0, 0, 0), "No engine", vk::makeApiVersion(0, 1, 0, 0), vk::makeApiVersion(0, 1, 0, 0) };

    auto extensions = GetRequiredExtensions(initInfo);
    vk::InstanceCreateInfo createInfo{
        vk::InstanceCreateFlags{},
        &appInfo,
        0, nullptr,                                                 // Validation layers.
        static_cast<uint32_t>(extensions.size()), extensions.data() // Extensions.
    };

    vk::DebugUtilsMessengerCreateInfoEXT debugCreateInfo{};
    if(_enableValidationLayers)
    {
        createInfo.enabledLayerCount = _validationLayers.size();
        createInfo.ppEnabledLayerNames = _validationLayers.data();

        util::PopulateDebugMessengerCreateInfo(debugCreateInfo);
        createInfo.pNext = static_cast<vk::DebugUtilsMessengerCreateInfoEXT*>(&debugCreateInfo);
    }
    else
    {
        createInfo.enabledLayerCount = 0;
        createInfo.pNext = nullptr;
    }

    util::VK_ASSERT(vk::createInstance(&createInfo, nullptr, &_instance), "Failed to create vk instance!");
}

bool Engine::CheckValidationLayerSupport()
{
    std::vector<vk::LayerProperties> availableLayers = vk::enumerateInstanceLayerProperties();
    bool result = std::all_of(_validationLayers.begin(), _validationLayers.end(), [&availableLayers](const auto& layerName)
    {
        const auto it = std::find_if(availableLayers.begin(), availableLayers.end(), [&layerName](const auto& layer){ return strcmp(layerName, layer.layerName) == 0; });

        return it != availableLayers.end();
    });

    return result;
}

void Engine::LogInstanceExtensions()
{
    std::vector<vk::ExtensionProperties> extensions = vk::enumerateInstanceExtensionProperties();

    std::cout << "Available extensions: \n";
    for(const auto& extension : extensions)
        std::cout << '\t' << extension.extensionName << '\n';
}

std::vector<const char*> Engine::GetRequiredExtensions(const InitInfo& initInfo)
{
    std::vector<const char*> extensions(initInfo.extensions, initInfo.extensions + initInfo.extensionCount);
    if(_enableValidationLayers)
        extensions.emplace_back(vk::EXTDebugUtilsExtensionName);

    return extensions;
}

void Engine::SetupDebugMessenger()
{
    if(!_enableValidationLayers)
        return;

    vk::DebugUtilsMessengerCreateInfoEXT createInfo{};
    util::PopulateDebugMessengerCreateInfo(createInfo);
    createInfo.pUserData = nullptr;

    util::VK_ASSERT(util::CreateDebugUtilsMessengerEXT(_instance, &createInfo, nullptr, &_debugMessenger), "Failed to create debug messenger!");
}

void Engine::PickPhysicalDevice()
{
    std::vector<vk::PhysicalDevice> devices = _instance.enumeratePhysicalDevices();
    if(devices.empty())
        throw std::runtime_error("No GPU's with Vulkan support available!");

    std::multimap<int, vk::PhysicalDevice> candidates{};

    for(const auto& device : devices)
    {
        uint32_t score = RateDeviceSuitability(device);
        if(score > 0)
            candidates.emplace(score, device);
    }
    if(candidates.empty())
        throw std::runtime_error("Failed finding suitable device!");

    _physicalDevice = candidates.rbegin()->second;
}

uint32_t Engine::RateDeviceSuitability(const vk::PhysicalDevice& device)
{
    vk::PhysicalDeviceProperties deviceProperties;
    vk::PhysicalDeviceFeatures deviceFeatures;
    device.getProperties(&deviceProperties);
    device.getFeatures(&deviceFeatures);

    QueueFamilyIndices familyIndices = FindQueueFamilies(device);

    uint32_t score{0};

    // Failed if geometry shader is not supported.
    if(!deviceFeatures.geometryShader)
        return 0;

    // Failed if graphics family queue is not supported.
    if(!familyIndices.IsComplete())
        return 0;

    // Failed if no extensions are supported.
    if(!ExtensionsSupported(device))
        return 0;

    // Check support for swap chain.
    SwapChainSupportDetails swapChainSupportDetails = QuerySwapChainSupport(device);
    bool swapChainUnsupported = swapChainSupportDetails.formats.empty() || swapChainSupportDetails.presentModes.empty();
    if(swapChainUnsupported)
        return 0;

    // Favor integrated GPUs above all else.
    if(deviceProperties.deviceType == vk::PhysicalDeviceType::eDiscreteGpu)
        score += 10000;

    score += deviceProperties.limits.maxImageDimension2D;

    return score;
}

bool Engine::ExtensionsSupported(const vk::PhysicalDevice& device)
{
    std::vector<vk::ExtensionProperties> availableExtensions = device.enumerateDeviceExtensionProperties();
    std::set<std::string> requiredExtensions{ _deviceExtensions.begin(), _deviceExtensions.end() };
    for(const auto& extension : availableExtensions)
        requiredExtensions.erase(extension.extensionName);

    return requiredExtensions.empty();
}

Engine::QueueFamilyIndices Engine::FindQueueFamilies(vk::PhysicalDevice const& device)
{
    QueueFamilyIndices indices{};

    uint32_t queueFamilyCount{0};
    device.getQueueFamilyProperties(&queueFamilyCount, nullptr);

    std::vector<vk::QueueFamilyProperties> queueFamilies(queueFamilyCount);
    device.getQueueFamilyProperties(&queueFamilyCount, queueFamilies.data());

    for(size_t i = 0; i < queueFamilies.size(); ++i)
    {
        if (queueFamilies[i].queueFlags & vk::QueueFlagBits::eGraphics)
            indices.graphicsFamily = i;

        vk::Bool32 supported;
        util::VK_ASSERT(device.getSurfaceSupportKHR(i, _surface, &supported), "Failed querying surface support on physical device!");
        if(supported)
            indices.presentFamily = i;

        if(indices.IsComplete())
            break;
    }

    return indices;
}

void Engine::CreateDevice()
{
    QueueFamilyIndices familyIndices = FindQueueFamilies(_physicalDevice);
    vk::PhysicalDeviceFeatures deviceFeatures;
    _physicalDevice.getFeatures(&deviceFeatures);

    std::vector<vk::DeviceQueueCreateInfo> queueCreateInfos{};
    std::set<uint32_t> uniqueQueueFamilies = { familyIndices.graphicsFamily.value(), familyIndices.presentFamily.value() };
    float queuePriority{ 1.0f };

    for(uint32_t familyQueueIndex : uniqueQueueFamilies)
        queueCreateInfos.emplace_back(vk::DeviceQueueCreateFlags{}, familyQueueIndex, 1, &queuePriority);

    vk::DeviceCreateInfo createInfo{};
    createInfo.queueCreateInfoCount = static_cast<uint32_t>(queueCreateInfos.size());
    createInfo.pQueueCreateInfos = queueCreateInfos.data();
    createInfo.pEnabledFeatures = &deviceFeatures;
    createInfo.enabledExtensionCount = static_cast<uint32_t>(_deviceExtensions.size());
    createInfo.ppEnabledExtensionNames = _deviceExtensions.data();

    if(_enableValidationLayers)
    {
        createInfo.enabledLayerCount = static_cast<uint32_t>(_validationLayers.size());
        createInfo.ppEnabledLayerNames = _validationLayers.data();
    }
    else
    {
        createInfo.enabledLayerCount = 0;
    }

    util::VK_ASSERT(_physicalDevice.createDevice(&createInfo, nullptr, &_device), "Failed creating a logical device!");

    _device.getQueue(familyIndices.graphicsFamily.value(), 0, &_graphicsQueue);
    _device.getQueue(familyIndices.presentFamily.value(), 0, &_presentQueue);
}

Engine::SwapChainSupportDetails Engine::QuerySwapChainSupport(const vk::PhysicalDevice& device)
{
    SwapChainSupportDetails details{};

    util::VK_ASSERT(device.getSurfaceCapabilitiesKHR(_surface, &details.capabilities), "Failed getting surface capabilities from physical device!");

    details.formats = device.getSurfaceFormatsKHR(_surface);
    details.presentModes = device.getSurfacePresentModesKHR(_surface);

    return details;
}

vk::SurfaceFormatKHR Engine::ChooseSwapSurfaceFormat(const std::vector<vk::SurfaceFormatKHR>& availableFormats)
{
    for(const auto& format : availableFormats)
    {
        if(format.format == vk::Format::eB8G8R8A8Srgb && format.colorSpace == vk::ColorSpaceKHR::eSrgbNonlinear)
            return format;
    }

    return availableFormats[0];
}

vk::PresentModeKHR Engine::ChoosePresentMode(const std::vector<vk::PresentModeKHR>& availablePresentModes)
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

vk::Extent2D Engine::ChooseSwapExtent(const vk::SurfaceCapabilitiesKHR& capabilities, const InitInfo& initInfo)
{
    if(capabilities.currentExtent.width != std::numeric_limits<uint32_t>::max())
        return capabilities.currentExtent;

    vk::Extent2D extent = { initInfo.width, initInfo.height };
    extent.width = std::clamp(extent.width, capabilities.minImageExtent.width, capabilities.maxImageExtent.width);
    extent.height = std::clamp(extent.height, capabilities.minImageExtent.height, capabilities.maxImageExtent.height);

    return extent;
}

void Engine::CreateSwapChain(const InitInfo& initInfo)
{
    SwapChainSupportDetails swapChainSupport = QuerySwapChainSupport(_physicalDevice);

    auto surfaceFormat = ChooseSwapSurfaceFormat(swapChainSupport.formats);
    auto presentMode = ChoosePresentMode(swapChainSupport.presentModes);
    auto extent = ChooseSwapExtent(swapChainSupport.capabilities, initInfo);

    uint32_t imageCount = swapChainSupport.capabilities.minImageCount + 1;
    if(swapChainSupport.capabilities.maxImageCount > 0 && imageCount > swapChainSupport.capabilities.maxImageCount)
        imageCount = swapChainSupport.capabilities.maxImageCount;

    vk::SwapchainCreateInfoKHR createInfo{};
    createInfo.surface = _surface;
    createInfo.minImageCount = imageCount;
    createInfo.imageFormat = surfaceFormat.format;
    createInfo.imageColorSpace = surfaceFormat.colorSpace;
    createInfo.imageExtent = extent;
    createInfo.imageArrayLayers = 1;
    createInfo.imageUsage = vk::ImageUsageFlagBits::eColorAttachment; // TODO: Can change this later to a memory transfer operation, when doing post-processing.

    QueueFamilyIndices indices = FindQueueFamilies(_physicalDevice);
    uint32_t queueFamilyIndices[] = { indices.graphicsFamily.value(), indices.presentFamily.value() };
    if(indices.graphicsFamily != indices.presentFamily)
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

    _swapChainImages = _device.getSwapchainImagesKHR(_swapChain);
    _swapChainFormat = surfaceFormat.format;
    _swapChainExtent = extent;
}

void Engine::CreateSwapChainImageViews()
{
    _swapChainImageViews.resize(_swapChainImages.size());
    for(size_t i = 0; i < _swapChainImageViews.size(); ++i)
    {
        vk::ImageViewCreateInfo createInfo{
            vk::ImageViewCreateFlags{},
            _swapChainImages[i],
            vk::ImageViewType::e2D,
            _swapChainFormat,
            vk::ComponentMapping{ vk::ComponentSwizzle::eIdentity, vk::ComponentSwizzle::eIdentity, vk::ComponentSwizzle::eIdentity, vk::ComponentSwizzle::eIdentity },
            vk::ImageSubresourceRange{
                vk::ImageAspectFlagBits::eColor, // aspect mask
                0, // base mip level
                1, // level count
                0, // base array level
                1  // layer count
            }
        };
        util::VK_ASSERT(_device.createImageView(&createInfo, nullptr, &_swapChainImageViews[i]), "Failed creating image view for swap chain!");
    }
}
