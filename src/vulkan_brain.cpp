#include "vulkan_brain.hpp"
#include "vulkan_helper.hpp"
#include "swap_chain.hpp"
#include "vulkan_validation.hpp"
#include <map>
#include <set>

VulkanBrain::VulkanBrain(const InitInfo& initInfo)
{
    CreateInstance(initInfo);
    dldi = vk::DispatchLoaderDynamic{ instance, vkGetInstanceProcAddr, device, vkGetDeviceProcAddr };
    SetupDebugMessenger();
    surface = initInfo.retrieveSurface(instance);
    PickPhysicalDevice();
    CreateDevice();

    CreateCommandPool();
    CreateDescriptorPool();

    VmaVulkanFunctions vulkanFunctions = {};
    vulkanFunctions.vkGetInstanceProcAddr = &vkGetInstanceProcAddr;
    vulkanFunctions.vkGetDeviceProcAddr = &vkGetDeviceProcAddr;

    VmaAllocatorCreateInfo vmaAllocatorCreateInfo{};
    vmaAllocatorCreateInfo.physicalDevice = physicalDevice;
    vmaAllocatorCreateInfo.device = device;
    vmaAllocatorCreateInfo.instance = instance;
    vmaAllocatorCreateInfo.vulkanApiVersion = VK_API_VERSION_1_3;
    vmaAllocatorCreateInfo.pVulkanFunctions = &vulkanFunctions;
    vmaCreateAllocator(&vmaAllocatorCreateInfo, &vmaAllocator);
}

VulkanBrain::~VulkanBrain()
{
    if(_enableValidationLayers)
        instance.destroyDebugUtilsMessengerEXT(_debugMessenger, nullptr, dldi);

    device.destroy(descriptorPool);

    device.destroy(commandPool);

    vmaDestroyAllocator(vmaAllocator);

    instance.destroy(surface);
    device.destroy();
    instance.destroy();
}

void VulkanBrain::CreateInstance(const InitInfo& initInfo)
{
    CheckValidationLayerSupport();
    if(_enableValidationLayers && !CheckValidationLayerSupport())
        throw std::runtime_error("Validation layers requested, but not supported!");

    vk::ApplicationInfo appInfo{};
    appInfo.pApplicationName = "";
    appInfo.applicationVersion = vk::makeApiVersion(0, 0, 0, 0);
    appInfo.engineVersion = vk::makeApiVersion(0, 1, 0, 0);
    appInfo.apiVersion = vk::makeApiVersion(0, 1, 1, 0);
    appInfo.pEngineName = "No engine";

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
        createInfo.pNext = &debugCreateInfo;
    } else
    {
        createInfo.enabledLayerCount = 0;
        createInfo.pNext = nullptr;
    }

    util::VK_ASSERT(vk::createInstance(&createInfo, nullptr, &instance), "Failed to create vk instance!");
}

void VulkanBrain::PickPhysicalDevice()
{
    std::vector<vk::PhysicalDevice> devices = instance.enumeratePhysicalDevices();
    if(devices.empty())
        throw std::runtime_error("No GPU's with Vulkan support available!");

    std::multimap<int, vk::PhysicalDevice> candidates{};

    for(const auto &device: devices)
    {
        uint32_t score = RateDeviceSuitability(device);
        if(score > 0)
            candidates.emplace(score, device);
    }
    if(candidates.empty())
        throw std::runtime_error("Failed finding suitable device!");

    physicalDevice = candidates.rbegin()->second;
}

uint32_t VulkanBrain::RateDeviceSuitability(const vk::PhysicalDevice& deviceToRate)
{
    vk::PhysicalDeviceProperties deviceProperties;
    vk::PhysicalDeviceFeatures deviceFeatures;
    deviceToRate.getProperties(&deviceProperties);
    deviceToRate.getFeatures(&deviceFeatures);

    QueueFamilyIndices familyIndices = QueueFamilyIndices::FindQueueFamilies(deviceToRate, surface);

    uint32_t score{ 0 };

    // Failed if geometry shader is not supported.
    if(!deviceFeatures.geometryShader)
        return 0;

    // Failed if graphics family queue is not supported.
    if(!familyIndices.IsComplete())
        return 0;

    // Failed if no extensions are supported.
    if(!ExtensionsSupported(deviceToRate))
        return 0;

    // Check support for swap chain.
    SwapChain::SupportDetails swapChainSupportDetails = SwapChain::QuerySupport(deviceToRate, surface);
    bool swapChainUnsupported = swapChainSupportDetails.formats.empty() || swapChainSupportDetails.presentModes.empty();
    if(swapChainUnsupported)
        return 0;

    // Favor discrete GPUs above all else.
    if(deviceProperties.deviceType == vk::PhysicalDeviceType::eDiscreteGpu)
        score += 50000;

    // Slightly favor integrated GPUs.
    if(deviceProperties.deviceType == vk::PhysicalDeviceType::eIntegratedGpu)
        score += 30000;

    score += deviceProperties.limits.maxImageDimension2D;

    return score;
}

bool VulkanBrain::ExtensionsSupported(const vk::PhysicalDevice &deviceToCheckSupport)
{
    std::vector<vk::ExtensionProperties> availableExtensions = deviceToCheckSupport.enumerateDeviceExtensionProperties();
    std::set<std::string> requiredExtensions{ _deviceExtensions.begin(), _deviceExtensions.end() };
    for(const auto &extension: availableExtensions)
        requiredExtensions.erase(extension.extensionName);

    return requiredExtensions.empty();
}

bool VulkanBrain::CheckValidationLayerSupport()
{
    std::vector<vk::LayerProperties> availableLayers = vk::enumerateInstanceLayerProperties();
    bool result = std::all_of(_validationLayers.begin(), _validationLayers.end(), [&availableLayers](const auto &layerName)
    {
        const auto it = std::find_if(availableLayers.begin(), availableLayers.end(), [&layerName](const auto &layer)
        { return strcmp(layerName, layer.layerName) == 0; });

        return it != availableLayers.end();
    });

    return result;
}

std::vector<const char*> VulkanBrain::GetRequiredExtensions(const InitInfo& initInfo)
{
    std::vector<const char *> extensions(initInfo.extensions, initInfo.extensions + initInfo.extensionCount);
    if(_enableValidationLayers)
        extensions.emplace_back(vk::EXTDebugUtilsExtensionName);

#if LINUX
    extensions.emplace_back(VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME);
#endif

    return extensions;
}

void VulkanBrain::SetupDebugMessenger()
{
    if(!_enableValidationLayers)
        return;

    vk::DebugUtilsMessengerCreateInfoEXT createInfo{};
    util::PopulateDebugMessengerCreateInfo(createInfo);
    createInfo.pUserData = nullptr;

    util::VK_ASSERT(instance.createDebugUtilsMessengerEXT( &createInfo, nullptr, &_debugMessenger, dldi),
                    "Failed to create debug messenger!");
}

void VulkanBrain::CreateDevice()
{
    queueFamilyIndices = QueueFamilyIndices::FindQueueFamilies(physicalDevice, surface);
    vk::PhysicalDeviceFeatures deviceFeatures;
    physicalDevice.getFeatures(&deviceFeatures);

    std::vector <vk::DeviceQueueCreateInfo> queueCreateInfos{};
    std::set <uint32_t> uniqueQueueFamilies = { queueFamilyIndices.graphicsFamily.value(), queueFamilyIndices.presentFamily.value() };
    float queuePriority{ 1.0f };

    for(uint32_t familyQueueIndex: uniqueQueueFamilies)
        queueCreateInfos.emplace_back(vk::DeviceQueueCreateFlags{}, familyQueueIndex, 1, &queuePriority);

    vk::PhysicalDeviceDynamicRenderingFeaturesKHR dynamicRenderingFeaturesKhr{};
    dynamicRenderingFeaturesKhr.dynamicRendering = true;

    vk::DeviceCreateInfo createInfo{};
    createInfo.pNext = &dynamicRenderingFeaturesKhr;
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

    util::VK_ASSERT(physicalDevice.createDevice(&createInfo, nullptr, &device), "Failed creating a logical device!");

    device.getQueue(queueFamilyIndices.graphicsFamily.value(), 0, &graphicsQueue);
    device.getQueue(queueFamilyIndices.presentFamily.value(), 0, &presentQueue);
}

void VulkanBrain::CreateCommandPool()
{
    vk::CommandPoolCreateInfo commandPoolCreateInfo{};
    commandPoolCreateInfo.flags = vk::CommandPoolCreateFlagBits::eResetCommandBuffer;
    commandPoolCreateInfo.queueFamilyIndex = queueFamilyIndices.graphicsFamily.value();

    util::VK_ASSERT(device.createCommandPool(&commandPoolCreateInfo, nullptr, &commandPool), "Failed creating command pool!");
}

void VulkanBrain::CreateDescriptorPool()
{
    std::vector<vk::DescriptorPoolSize> poolSizes = {
            { vk::DescriptorType::eSampler,              1000 },
            { vk::DescriptorType::eCombinedImageSampler, 1000 },
            { vk::DescriptorType::eSampledImage,         1000 },
            { vk::DescriptorType::eStorageImage,         1000 },
            { vk::DescriptorType::eUniformTexelBuffer,   1000 },
            { vk::DescriptorType::eStorageTexelBuffer,   1000 },
            { vk::DescriptorType::eUniformBuffer,        1000 },
            { vk::DescriptorType::eStorageBuffer,        1000 },
            { vk::DescriptorType::eUniformBufferDynamic, 1000 },
            { vk::DescriptorType::eStorageBufferDynamic, 1000 },
            { vk::DescriptorType::eInputAttachment,      1000 }
    };

    vk::DescriptorPoolCreateInfo createInfo{};
    createInfo.poolSizeCount = poolSizes.size();
    createInfo.pPoolSizes = poolSizes.data();
    createInfo.maxSets = 200;
    createInfo.flags = vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet;
    util::VK_ASSERT(device.createDescriptorPool(&createInfo, nullptr, &descriptorPool), "Failed creating descriptor pool!");
}

QueueFamilyIndices QueueFamilyIndices::FindQueueFamilies(vk::PhysicalDevice device, vk::SurfaceKHR surface)
{
    QueueFamilyIndices indices{};

    uint32_t queueFamilyCount{ 0 };
    device.getQueueFamilyProperties(&queueFamilyCount, nullptr);

    std::vector <vk::QueueFamilyProperties> queueFamilies(queueFamilyCount);
    device.getQueueFamilyProperties(&queueFamilyCount, queueFamilies.data());

    for(size_t i = 0; i < queueFamilies.size(); ++i)
    {
        if(queueFamilies[i].queueFlags & vk::QueueFlagBits::eGraphics)
            indices.graphicsFamily = i;

        if(!indices.presentFamily.has_value())
        {
            vk::Bool32 supported;
            util::VK_ASSERT(device.getSurfaceSupportKHR(i, surface, &supported),
                            "Failed querying surface support on physical device!");
            if(supported)
                indices.presentFamily = i;
        }

        if(indices.IsComplete())
            break;
    }

    return indices;
}
