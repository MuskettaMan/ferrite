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
}

void Engine::Run()
{

}

void Engine::Shutdown()
{
    if(_enableValidationLayers)
        util::DestroyDebugUtilsMessengerEXT(_instance, _debugMessenger, nullptr);

    // TODO: Find nicer way to destroy surface.
    vkDestroySurfaceKHR(_instance, _surface, nullptr);
    _device.destroy();
    _instance.destroy();
}

void Engine::CreateInstance(const InitInfo& initInfo)
{
    if(_enableValidationLayers && !CheckValidationLayerSupport())
        throw std::runtime_error("Validation layers requested, but not supported!");

    vk::ApplicationInfo appInfo{ "", vk::makeApiVersion(0, 0, 0, 0), "No engine", vk::makeApiVersion(0, 1, 0, 0), vk::makeApiVersion(0, 1, 0, 0) };

    auto extensions = GetRequiredExtensions(initInfo);
    vk::InstanceCreateInfo createInfo{
        vk::InstanceCreateFlags{},
        &appInfo,
        static_cast<uint32_t>(_validationLayers.size()), _validationLayers.data(),
        static_cast<uint32_t>(extensions.size()), extensions.data()
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
    uint32_t layerCount{};
    util::VK_ASSERT(vk::enumerateInstanceLayerProperties(&layerCount, nullptr), "Failed to enumerate instance layer properties!");

    std::vector<vk::LayerProperties> availableLayers(layerCount);
    util::VK_ASSERT(vk::enumerateInstanceLayerProperties(&layerCount, availableLayers.data()), "Failed to enumerate instance layer properties!");

    bool result = std::all_of(_validationLayers.begin(), _validationLayers.end(), [&availableLayers](const auto& layerName)
    {
        const auto it = std::find_if(availableLayers.begin(), availableLayers.end(), [&layerName](const auto& layer){ return strcmp(layerName, layer.layerName) == 0; });

        return it != availableLayers.end();
    });

    return result;
}

void Engine::LogInstanceExtensions()
{
    uint32_t extensionCount{ 0 };
    util::VK_ASSERT(vk::enumerateInstanceExtensionProperties(nullptr, &extensionCount, nullptr), "Failed to enumerate instance extension properties!");

    std::vector<vk::ExtensionProperties> extensions(extensionCount);
    util::VK_ASSERT(vk::enumerateInstanceExtensionProperties(nullptr, &extensionCount, extensions.data()), "Failed to enumerate instance extension properties!");
    std::cout << "Available extensions: \n";
    for(const auto& extension : extensions)
    {
        std::cout << '\t' << extension.extensionName << '\n';
    }
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
    uint32_t deviceCount{0};
    util::VK_ASSERT(_instance.enumeratePhysicalDevices(&deviceCount, nullptr), "Failed to enumerate physical devices!");

    if(deviceCount == 0)
        throw std::runtime_error("No GPU's with Vulkan support available!");

    std::vector<vk::PhysicalDevice> devices(deviceCount);
    util::VK_ASSERT(_instance.enumeratePhysicalDevices(&deviceCount, devices.data()), "Failed to enumerate physical devices!");

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

    // Favor integrated GPUs above all else.
    if(deviceProperties.deviceType == vk::PhysicalDeviceType::eDiscreteGpu)
        score += 10000;

    score += deviceProperties.limits.maxImageDimension2D;

    return score;
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
    createInfo.enabledExtensionCount = 0;
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
