#include <iostream>
#include <cstdint>
#include <vector>
#include <algorithm>
#include <cstring>
#include <map>
#include <set>
#include <implot.h>
#include <thread>

#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

#define VMA_IMPLEMENTATION
#include "vk_mem_alloc.h"

#include "imgui.h"
#include "engine.hpp"
#include "vulkan_validation.hpp"
#include "vulkan_helper.hpp"
#include "shaders/shader_loader.hpp"
#include "imgui_impl_vulkan.h"
#include "stopwatch.hpp"
#include "model_loader.hpp"
#include "util.hpp"
#include "spdlog/spdlog.h"

Engine::Engine()
{
    ImGui::CreateContext();
    ImPlot::CreateContext();
}

void Engine::Init(const InitInfo &initInfo, std::shared_ptr<Application> application)
{
    _application = std::move(application);

    CreateInstance(initInfo);
    _dldi = vk::DispatchLoaderDynamic{ _instance, vkGetInstanceProcAddr, _device, vkGetDeviceProcAddr };

    SetupDebugMessenger();

    _surface = initInfo.retrieveSurface(_instance);

    PickPhysicalDevice();
    CreateDevice();
    CreateCommandPool();

    VmaVulkanFunctions vulkanFunctions = {};
    vulkanFunctions.vkGetInstanceProcAddr = &vkGetInstanceProcAddr;
    vulkanFunctions.vkGetDeviceProcAddr = &vkGetDeviceProcAddr;

    VmaAllocatorCreateInfo vmaAllocatorCreateInfo{};
    vmaAllocatorCreateInfo.physicalDevice = _physicalDevice;
    vmaAllocatorCreateInfo.device = _device;
    vmaAllocatorCreateInfo.instance = _instance;
    vmaAllocatorCreateInfo.vulkanApiVersion = VK_API_VERSION_1_3;
    vmaAllocatorCreateInfo.pVulkanFunctions = &vulkanFunctions;
    vmaCreateAllocator(&vmaAllocatorCreateInfo, &_vmaAllocator);


    auto supportedDepthFormat = util::FindSupportedFormat(_physicalDevice, { vk::Format::eD32Sfloat, vk::Format::eD32SfloatS8Uint, vk::Format::eD24UnormS8Uint },
                                               vk::ImageTiling::eOptimal,
                                               vk::FormatFeatureFlagBits::eDepthStencilAttachment);

    assert(supportedDepthFormat.has_value() && "No supported depth format!");

    _swapChain = std::make_unique<SwapChain>(_device, _physicalDevice, _instance, _commandPool, _vmaAllocator, _graphicsQueue, _surface, supportedDepthFormat.value());

    QueueFamilyIndices familyIndices = FindQueueFamilies(_physicalDevice);
    _swapChain->CreateSwapChain(glm::uvec2{ initInfo.width, initInfo.height }, familyIndices);


    CreateDescriptorSetLayout();
    InitializeDeferredRTs();

    CreateGeometryPipeline();
    CreateLightingPipeline();

    CreateTextureSampler();

    CreateUniformBuffers();

    CreateCommandBuffers();
    CreateSyncObjects();
    CreateDescriptorPool();


    ModelLoader modelLoader;
    Model model = modelLoader.Load("assets/models/DamagedHelmet.glb");
    _model = LoadModel(model);

    CreateDescriptorSets();

    _newImGuiFrame = initInfo.newImGuiFrame;
    _shutdownImGui = initInfo.shutdownImGui;

    vk::Format format = _swapChain->GetFormat();
    vk::PipelineRenderingCreateInfoKHR pipelineRenderingCreateInfoKhr{};
    pipelineRenderingCreateInfoKhr.colorAttachmentCount = 1;
    pipelineRenderingCreateInfoKhr.pColorAttachmentFormats = &format;
    pipelineRenderingCreateInfoKhr.depthAttachmentFormat = _swapChain->GetDepthFormat();

    ImGui_ImplVulkan_InitInfo initInfoVulkan{};
    initInfoVulkan.UseDynamicRendering = true;
    initInfoVulkan.PipelineRenderingCreateInfo = static_cast<VkPipelineRenderingCreateInfo>(pipelineRenderingCreateInfoKhr);
    initInfoVulkan.PhysicalDevice = _physicalDevice;
    initInfoVulkan.Device = _device;
    initInfoVulkan.ImageCount = MAX_FRAMES_IN_FLIGHT;
    initInfoVulkan.Instance = _instance;
    initInfoVulkan.MSAASamples = VkSampleCountFlagBits::VK_SAMPLE_COUNT_1_BIT;
    initInfoVulkan.Queue = _graphicsQueue;
    initInfoVulkan.QueueFamily = familyIndices.graphicsFamily.value();
    initInfoVulkan.DescriptorPool = _descriptorPool;
    initInfoVulkan.MinImageCount = 2;
    initInfoVulkan.ImageCount = _swapChain->GetImageCount();
    ImGui_ImplVulkan_Init(&initInfoVulkan);

    ImGui_ImplVulkan_CreateFontsTexture();
}

void Engine::Run()
{
    // Slow down application when minimized.
    if(_application->IsMinimized())
    {
        using namespace std::chrono_literals;
        std::this_thread::sleep_for(16ms);
        return;
    }

    static std::vector<PerformanceTracker::FrameData> frameData;
    frameData.clear();

    static Stopwatch stopwatch{};
    stopwatch.start();

    util::VK_ASSERT(_device.waitForFences(1, &_inFlightFences[_currentFrame], vk::True, std::numeric_limits<uint64_t>::max()),
                    "Failed waiting on in flight fence!");

    stopwatch.stop();

    frameData.emplace_back("Fence duration", stopwatch.elapsed_milliseconds());

    stopwatch.start();

    uint32_t imageIndex;
    vk::Result result = _device.acquireNextImageKHR(_swapChain->GetSwapChain(), std::numeric_limits<uint64_t>::max(),
                                                    _imageAvailableSemaphores[_currentFrame], nullptr, &imageIndex);

    stopwatch.stop();

    frameData.emplace_back("Acquiring next image", stopwatch.elapsed_milliseconds());

    if(result == vk::Result::eErrorOutOfDateKHR || result == vk::Result::eSuboptimalKHR)
    {
        QueueFamilyIndices familyIndices = FindQueueFamilies(_physicalDevice);
        _swapChain->RecreateSwapChain(_application->DisplaySize(), familyIndices);

        return;
    } else
        util::VK_ASSERT(result, "Failed acquiring next image from swap chain!");

    util::VK_ASSERT(_device.resetFences(1, &_inFlightFences[_currentFrame]), "Failed resetting fences!");

    UpdateUniformData(_currentFrame);

    stopwatch.start();

    ImGui_ImplVulkan_NewFrame();
    _newImGuiFrame();
    ImGui::NewFrame();

    // ImGui stuff
    _performanceTracker.Render();

    ImGui::Render();

    _commandBuffers[_currentFrame].reset();
    stopwatch.stop();

    frameData.emplace_back("Rendering ImGui", stopwatch.elapsed_milliseconds());

    stopwatch.start();
    RecordCommandBuffer(_commandBuffers[_currentFrame], imageIndex);
    stopwatch.stop();

    frameData.emplace_back("Recording command buffer", stopwatch.elapsed_milliseconds());


    vk::SubmitInfo submitInfo{};
    vk::Semaphore waitSemaphores[] = { _imageAvailableSemaphores[_currentFrame] };
    vk::PipelineStageFlags waitStages[] = { vk::PipelineStageFlagBits::eColorAttachmentOutput };
    submitInfo.waitSemaphoreCount = 1;
    submitInfo.pWaitSemaphores = waitSemaphores;
    submitInfo.pWaitDstStageMask = waitStages;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &_commandBuffers[_currentFrame];

    vk::Semaphore signalSemaphores[] = { _renderFinishedSemaphores[_currentFrame] };
    submitInfo.signalSemaphoreCount = 1;
    submitInfo.pSignalSemaphores = signalSemaphores;

    stopwatch.start();
    util::VK_ASSERT(_graphicsQueue.submit(1, &submitInfo, _inFlightFences[_currentFrame]), "Failed submitting to graphics queue!");
    stopwatch.stop();
    frameData.emplace_back("Submit to graphics queue", stopwatch.elapsed_milliseconds());


    vk::PresentInfoKHR presentInfo{};
    presentInfo.waitSemaphoreCount = 1;
    presentInfo.pWaitSemaphores = signalSemaphores;

    vk::SwapchainKHR swapchains[] = { _swapChain->GetSwapChain() };
    presentInfo.swapchainCount = 1;
    presentInfo.pSwapchains = swapchains;
    presentInfo.pImageIndices = &imageIndex;

    stopwatch.start();
    result = _presentQueue.presentKHR(&presentInfo);
    stopwatch.stop();

    frameData.emplace_back("Presenting", stopwatch.elapsed_milliseconds());

    stopwatch.start();
    _device.waitIdle();
    stopwatch.stop();
    frameData.emplace_back("Wait idle", stopwatch.elapsed_milliseconds());

    if(result == vk::Result::eErrorOutOfDateKHR || result == vk::Result::eSuboptimalKHR || _swapChain->GetImageSize() != _application->DisplaySize())
    {
        _swapChain->RecreateSwapChain(_application->DisplaySize(), _queueFamilyIndices);
    }
    else
    {
        util::VK_ASSERT(result, "Failed acquiring next image from swap chain!");
    }

    _currentFrame = (_currentFrame + 1) % MAX_FRAMES_IN_FLIGHT;

    _performanceTracker.Update(frameData);
}

void Engine::Shutdown()
{
    _shutdownImGui();
    ImGui_ImplVulkan_Shutdown();
    ImPlot::DestroyContext();
    ImGui::DestroyContext();

    for(size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i)
    {
        _device.destroy(_inFlightFences[i]);
        _device.destroy(_renderFinishedSemaphores[i]);
        _device.destroy(_imageAvailableSemaphores[i]);
    }

    if(_enableValidationLayers)
        _instance.destroyDebugUtilsMessengerEXT(_debugMessenger, nullptr, _dldi);


    for(auto& mesh : _model.meshes)
    {
        for(auto& primitive : mesh.primitives)
        {
            _device.destroy(primitive.vertexBuffer);
            _device.destroy(primitive.indexBuffer);
            vmaFreeMemory(_vmaAllocator, primitive.vertexBufferAllocation);
            vmaFreeMemory(_vmaAllocator, primitive.indexBufferAllocation);
        }
    }
    for(auto& texture : _model.textures)
    {
        _device.destroy(texture.image);
        _device.destroy(texture.imageView);
        vmaFreeMemory(_vmaAllocator, texture.imageAllocation);
    }

    _swapChain.reset();

    _device.destroy(_descriptorPool);

    _device.destroy(_commandPool);

    _device.destroy(_sampler);

    _device.destroy(_geometryPipeline);
    _device.destroy(_geometryPipelineLayout);
    _device.destroy(_lightingPipeline);
    _device.destroy(_lightingPipelineLayout);

    for(size_t i = 0; i < _frameData.size(); ++i)
    {
        _device.destroy(_frameData[i].uniformBuffer);
        vmaUnmapMemory(_vmaAllocator, _frameData[i].uniformBufferAllocation);
        vmaFreeMemory(_vmaAllocator, _frameData[i].uniformBufferAllocation);

        for(size_t j = 0; j < _frameData[i].deferredAttachments.size(); ++j)
        {
            _device.destroy(_frameData[i].deferredAttachments[j].image);
            _device.destroy(_frameData[i].deferredAttachments[j].imageView);
            vmaFreeMemory(_vmaAllocator, _frameData[i].deferredAttachments[j].imageAllocation);
        }
    }

    _device.destroy(_geometryDescriptorSetLayout);
    _device.destroy(_lightingDescriptorSetLayout);

    vmaDestroyAllocator(_vmaAllocator);

    _instance.destroy(_surface);
    _device.destroy();
    _instance.destroy();
}

void Engine::CreateInstance(const InitInfo &initInfo)
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

    util::VK_ASSERT(vk::createInstance(&createInfo, nullptr, &_instance), "Failed to create vk instance!");
}

bool Engine::CheckValidationLayerSupport()
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

std::vector<const char *> Engine::GetRequiredExtensions(const InitInfo &initInfo)
{
    std::vector<const char *> extensions(initInfo.extensions, initInfo.extensions + initInfo.extensionCount);
    if(_enableValidationLayers)
        extensions.emplace_back(vk::EXTDebugUtilsExtensionName);

#if LINUX
    extensions.emplace_back(VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME);
#endif

    return extensions;
}

void Engine::SetupDebugMessenger()
{
    if(!_enableValidationLayers)
        return;

    vk::DebugUtilsMessengerCreateInfoEXT createInfo{};
    util::PopulateDebugMessengerCreateInfo(createInfo);
    createInfo.pUserData = nullptr;

    util::VK_ASSERT(_instance.createDebugUtilsMessengerEXT( &createInfo, nullptr, &_debugMessenger, _dldi),
                    "Failed to create debug messenger!");
}

void Engine::PickPhysicalDevice()
{
    std::vector<vk::PhysicalDevice> devices = _instance.enumeratePhysicalDevices();
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

    _physicalDevice = candidates.rbegin()->second;
}

uint32_t Engine::RateDeviceSuitability(const vk::PhysicalDevice &device)
{
    vk::PhysicalDeviceProperties deviceProperties;
    vk::PhysicalDeviceFeatures deviceFeatures;
    device.getProperties(&deviceProperties);
    device.getFeatures(&deviceFeatures);

    QueueFamilyIndices familyIndices = FindQueueFamilies(device);

    uint32_t score{ 0 };

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
    SwapChain::SupportDetails swapChainSupportDetails = SwapChain::QuerySupport(device, _surface);
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

bool Engine::ExtensionsSupported(const vk::PhysicalDevice &device)
{
    std::vector<vk::ExtensionProperties> availableExtensions = device.enumerateDeviceExtensionProperties();
    std::set<std::string> requiredExtensions{ _deviceExtensions.begin(), _deviceExtensions.end() };
    for(const auto &extension: availableExtensions)
        requiredExtensions.erase(extension.extensionName);

    return requiredExtensions.empty();
}

QueueFamilyIndices Engine::FindQueueFamilies(vk::PhysicalDevice const &device)
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
            util::VK_ASSERT(device.getSurfaceSupportKHR(i, _surface, &supported),
                            "Failed querying surface support on physical device!");
            if(supported)
                indices.presentFamily = i;
        }

        if(indices.IsComplete())
            break;
    }

    return indices;
}

void Engine::CreateDevice()
{
    _queueFamilyIndices = FindQueueFamilies(_physicalDevice);
    vk::PhysicalDeviceFeatures deviceFeatures;
    _physicalDevice.getFeatures(&deviceFeatures);

    std::vector <vk::DeviceQueueCreateInfo> queueCreateInfos{};
    std::set <uint32_t> uniqueQueueFamilies = { _queueFamilyIndices.graphicsFamily.value(), _queueFamilyIndices.presentFamily.value() };
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

    util::VK_ASSERT(_physicalDevice.createDevice(&createInfo, nullptr, &_device), "Failed creating a logical device!");

    _device.getQueue(_queueFamilyIndices.graphicsFamily.value(), 0, &_graphicsQueue);
    _device.getQueue(_queueFamilyIndices.presentFamily.value(), 0, &_presentQueue);
}

void Engine::CreateGeometryPipeline()
{
    vk::PipelineLayoutCreateInfo pipelineLayoutCreateInfo{};
    pipelineLayoutCreateInfo.setLayoutCount = 1;
    pipelineLayoutCreateInfo.pSetLayouts = &_geometryDescriptorSetLayout;
    pipelineLayoutCreateInfo.pushConstantRangeCount = 0;
    pipelineLayoutCreateInfo.pPushConstantRanges = nullptr;

    util::VK_ASSERT(_device.createPipelineLayout(&pipelineLayoutCreateInfo, nullptr, &_geometryPipelineLayout),
                    "Failed creating geometry pipeline layout!");

    auto vertByteCode = shader::ReadFile("shaders/geom-v.spv");
    auto fragByteCode = shader::ReadFile("shaders/geom-f.spv");

    vk::ShaderModule vertModule = shader::CreateShaderModule(vertByteCode, _device);
    vk::ShaderModule fragModule = shader::CreateShaderModule(fragByteCode, _device);

    vk::PipelineShaderStageCreateInfo vertShaderStageCreateInfo{};
    vertShaderStageCreateInfo.stage = vk::ShaderStageFlagBits::eVertex;
    vertShaderStageCreateInfo.module = vertModule;
    vertShaderStageCreateInfo.pName = "main";

    vk::PipelineShaderStageCreateInfo fragShaderStageCreateInfo{};
    fragShaderStageCreateInfo.stage = vk::ShaderStageFlagBits::eFragment;
    fragShaderStageCreateInfo.module = fragModule;
    fragShaderStageCreateInfo.pName = "main";

    vk::PipelineShaderStageCreateInfo shaderStages[] = { vertShaderStageCreateInfo, fragShaderStageCreateInfo };

    auto bindingDesc = Vertex::GetBindingDescription();
    auto attributes = Vertex::GetAttributeDescriptions();

    vk::PipelineVertexInputStateCreateInfo vertexInputStateCreateInfo{};
    vertexInputStateCreateInfo.vertexBindingDescriptionCount = 1;
    vertexInputStateCreateInfo.pVertexBindingDescriptions = &bindingDesc;
    vertexInputStateCreateInfo.vertexAttributeDescriptionCount = attributes.size();
    vertexInputStateCreateInfo.pVertexAttributeDescriptions = attributes.data();

    vk::PipelineInputAssemblyStateCreateInfo inputAssemblyStateCreateInfo{};
    inputAssemblyStateCreateInfo.topology = vk::PrimitiveTopology::eTriangleList;
    inputAssemblyStateCreateInfo.primitiveRestartEnable = vk::False;

    vk::Extent2D swapChainExtent{ _swapChain->GetExtent() };
    _viewport = vk::Viewport{ 0.0f, 0.0f, static_cast<float>(swapChainExtent.width), static_cast<float>(swapChainExtent.height), 0.0f,
                              1.0f };
    _scissor = vk::Rect2D{ vk::Offset2D{ 0, 0 }, swapChainExtent };

    std::array<vk::DynamicState, 2> dynamicStates = {
            vk::DynamicState::eViewport,
            vk::DynamicState::eScissor,
    };

    vk::PipelineDynamicStateCreateInfo dynamicStateCreateInfo{};
    dynamicStateCreateInfo.dynamicStateCount = dynamicStates.size();
    dynamicStateCreateInfo.pDynamicStates = dynamicStates.data();

    vk::PipelineViewportStateCreateInfo viewportStateCreateInfo{};
    viewportStateCreateInfo.viewportCount = 1;
    viewportStateCreateInfo.scissorCount = 1;

    vk::PipelineRasterizationStateCreateInfo rasterizationStateCreateInfo{};
    rasterizationStateCreateInfo.depthClampEnable = vk::False;
    rasterizationStateCreateInfo.rasterizerDiscardEnable = vk::False;
    rasterizationStateCreateInfo.polygonMode = vk::PolygonMode::eFill;
    rasterizationStateCreateInfo.lineWidth = 1.0f;
    rasterizationStateCreateInfo.cullMode = vk::CullModeFlagBits::eBack;
    rasterizationStateCreateInfo.frontFace = vk::FrontFace::eCounterClockwise;
    rasterizationStateCreateInfo.depthBiasEnable = vk::False;
    rasterizationStateCreateInfo.depthBiasConstantFactor = 0.0f;
    rasterizationStateCreateInfo.depthBiasClamp = 0.0f;
    rasterizationStateCreateInfo.depthBiasSlopeFactor = 0.0f;

    vk::PipelineMultisampleStateCreateInfo multisampleStateCreateInfo{};
    multisampleStateCreateInfo.sampleShadingEnable = vk::False;
    multisampleStateCreateInfo.rasterizationSamples = vk::SampleCountFlagBits::e1;
    multisampleStateCreateInfo.minSampleShading = 1.0f;
    multisampleStateCreateInfo.pSampleMask = nullptr;
    multisampleStateCreateInfo.alphaToCoverageEnable = vk::False;
    multisampleStateCreateInfo.alphaToOneEnable = vk::False;

    std::array<vk::PipelineColorBlendAttachmentState, 3> colorBlendAttachmentStates{};
    for(auto& blendAttachmentState : colorBlendAttachmentStates)
    {
        blendAttachmentState.blendEnable = vk::False;
        blendAttachmentState.colorWriteMask =
                vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG | vk::ColorComponentFlagBits::eB |
                vk::ColorComponentFlagBits::eA;
    }

    vk::PipelineColorBlendStateCreateInfo colorBlendStateCreateInfo{};
    colorBlendStateCreateInfo.logicOpEnable = vk::False;
    colorBlendStateCreateInfo.attachmentCount = colorBlendAttachmentStates.size();
    colorBlendStateCreateInfo.pAttachments = colorBlendAttachmentStates.data();

    vk::PipelineDepthStencilStateCreateInfo depthStencilStateCreateInfo{};
    depthStencilStateCreateInfo.depthTestEnable = true;
    depthStencilStateCreateInfo.depthWriteEnable = true;
    depthStencilStateCreateInfo.depthCompareOp = vk::CompareOp::eLess;
    depthStencilStateCreateInfo.depthBoundsTestEnable = false;
    depthStencilStateCreateInfo.minDepthBounds = 0.0f;
    depthStencilStateCreateInfo.maxDepthBounds = 1.0f;
    depthStencilStateCreateInfo.stencilTestEnable = false;

    vk::GraphicsPipelineCreateInfo geometryPipelineCreateInfo{};
    geometryPipelineCreateInfo.stageCount = 2;
    geometryPipelineCreateInfo.pStages = shaderStages;
    geometryPipelineCreateInfo.pVertexInputState = &vertexInputStateCreateInfo;
    geometryPipelineCreateInfo.pInputAssemblyState = &inputAssemblyStateCreateInfo;
    geometryPipelineCreateInfo.pViewportState = &viewportStateCreateInfo;
    geometryPipelineCreateInfo.pRasterizationState = &rasterizationStateCreateInfo;
    geometryPipelineCreateInfo.pMultisampleState = &multisampleStateCreateInfo;
    geometryPipelineCreateInfo.pDepthStencilState = &depthStencilStateCreateInfo;
    geometryPipelineCreateInfo.pColorBlendState = &colorBlendStateCreateInfo;
    geometryPipelineCreateInfo.pDynamicState = &dynamicStateCreateInfo;
    geometryPipelineCreateInfo.layout = _geometryPipelineLayout;
    geometryPipelineCreateInfo.subpass = 0;
    geometryPipelineCreateInfo.basePipelineHandle = nullptr;
    geometryPipelineCreateInfo.basePipelineIndex = -1;

    vk::PipelineRenderingCreateInfoKHR pipelineRenderingCreateInfoKhr{};
    std::array<vk::Format, 3> formats{ vk::Format::eR16G16B16A16Sfloat, vk::Format::eR16G16B16A16Sfloat, vk::Format::eR16G16B16A16Sfloat };
    pipelineRenderingCreateInfoKhr.colorAttachmentCount = FrameData::DEFERRED_ATTACHMENT_COUNT;
    pipelineRenderingCreateInfoKhr.pColorAttachmentFormats = formats.data();
    pipelineRenderingCreateInfoKhr.depthAttachmentFormat = _swapChain->GetDepthFormat();

    geometryPipelineCreateInfo.pNext = &pipelineRenderingCreateInfoKhr;
    geometryPipelineCreateInfo.renderPass = nullptr; // Using dynamic rendering.

    auto result = _device.createGraphicsPipeline(nullptr, geometryPipelineCreateInfo, nullptr);
    util::VK_ASSERT(result.result, "Failed creating the geometry pipeline layout!");
    _geometryPipeline = result.value;

    _device.destroy(vertModule);
    _device.destroy(fragModule);
}

void Engine::CreateLightingPipeline()
{
    vk::PipelineLayoutCreateInfo pipelineLayoutCreateInfo{};
    pipelineLayoutCreateInfo.setLayoutCount = 1;
    pipelineLayoutCreateInfo.pSetLayouts = &_lightingDescriptorSetLayout;
    pipelineLayoutCreateInfo.pushConstantRangeCount = 0;
    pipelineLayoutCreateInfo.pPushConstantRanges = nullptr;

    util::VK_ASSERT(_device.createPipelineLayout(&pipelineLayoutCreateInfo, nullptr, &_lightingPipelineLayout),
                    "Failed creating geometry pipeline layout!");

    auto vertByteCode = shader::ReadFile("shaders/lighting-v.spv");
    auto fragByteCode = shader::ReadFile("shaders/lighting-f.spv");

    vk::ShaderModule vertModule = shader::CreateShaderModule(vertByteCode, _device);
    vk::ShaderModule fragModule = shader::CreateShaderModule(fragByteCode, _device);

    vk::PipelineShaderStageCreateInfo vertShaderStageCreateInfo{};
    vertShaderStageCreateInfo.stage = vk::ShaderStageFlagBits::eVertex;
    vertShaderStageCreateInfo.module = vertModule;
    vertShaderStageCreateInfo.pName = "main";

    vk::PipelineShaderStageCreateInfo fragShaderStageCreateInfo{};
    fragShaderStageCreateInfo.stage = vk::ShaderStageFlagBits::eFragment;
    fragShaderStageCreateInfo.module = fragModule;
    fragShaderStageCreateInfo.pName = "main";

    vk::PipelineShaderStageCreateInfo shaderStages[] = { vertShaderStageCreateInfo, fragShaderStageCreateInfo };

    vk::PipelineVertexInputStateCreateInfo vertexInputStateCreateInfo{};

    vk::PipelineInputAssemblyStateCreateInfo inputAssemblyStateCreateInfo{};
    inputAssemblyStateCreateInfo.topology = vk::PrimitiveTopology::eTriangleList;
    inputAssemblyStateCreateInfo.primitiveRestartEnable = vk::False;

    vk::Extent2D swapChainExtent{ _swapChain->GetExtent() };
    _viewport = vk::Viewport{ 0.0f, 0.0f, static_cast<float>(swapChainExtent.width), static_cast<float>(swapChainExtent.height), 0.0f,
                              1.0f };
    _scissor = vk::Rect2D{ vk::Offset2D{ 0, 0 }, swapChainExtent };

    std::array<vk::DynamicState, 2> dynamicStates = {
            vk::DynamicState::eViewport,
            vk::DynamicState::eScissor,
    };

    vk::PipelineDynamicStateCreateInfo dynamicStateCreateInfo{};
    dynamicStateCreateInfo.dynamicStateCount = dynamicStates.size();
    dynamicStateCreateInfo.pDynamicStates = dynamicStates.data();

    vk::PipelineViewportStateCreateInfo viewportStateCreateInfo{};
    viewportStateCreateInfo.viewportCount = 1;
    viewportStateCreateInfo.scissorCount = 1;

    vk::PipelineRasterizationStateCreateInfo rasterizationStateCreateInfo{};
    rasterizationStateCreateInfo.depthClampEnable = vk::False;
    rasterizationStateCreateInfo.rasterizerDiscardEnable = vk::False;
    rasterizationStateCreateInfo.polygonMode = vk::PolygonMode::eFill;
    rasterizationStateCreateInfo.lineWidth = 1.0f;
    rasterizationStateCreateInfo.cullMode = vk::CullModeFlagBits::eBack;
    rasterizationStateCreateInfo.frontFace = vk::FrontFace::eClockwise;
    rasterizationStateCreateInfo.depthBiasEnable = vk::False;
    rasterizationStateCreateInfo.depthBiasConstantFactor = 0.0f;
    rasterizationStateCreateInfo.depthBiasClamp = 0.0f;
    rasterizationStateCreateInfo.depthBiasSlopeFactor = 0.0f;

    vk::PipelineMultisampleStateCreateInfo multisampleStateCreateInfo{};
    multisampleStateCreateInfo.sampleShadingEnable = vk::False;
    multisampleStateCreateInfo.rasterizationSamples = vk::SampleCountFlagBits::e1;
    multisampleStateCreateInfo.minSampleShading = 1.0f;
    multisampleStateCreateInfo.pSampleMask = nullptr;
    multisampleStateCreateInfo.alphaToCoverageEnable = vk::False;
    multisampleStateCreateInfo.alphaToOneEnable = vk::False;

    vk::PipelineColorBlendAttachmentState colorBlendAttachmentState{};
    colorBlendAttachmentState.blendEnable = vk::False;
    colorBlendAttachmentState.colorWriteMask =
            vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG | vk::ColorComponentFlagBits::eB |
            vk::ColorComponentFlagBits::eA;

    vk::PipelineColorBlendStateCreateInfo colorBlendStateCreateInfo{};
    colorBlendStateCreateInfo.logicOpEnable = vk::False;
    colorBlendStateCreateInfo.attachmentCount = 1;
    colorBlendStateCreateInfo.pAttachments = &colorBlendAttachmentState;

    vk::PipelineDepthStencilStateCreateInfo depthStencilStateCreateInfo{};
    depthStencilStateCreateInfo.depthTestEnable = false;
    depthStencilStateCreateInfo.depthWriteEnable = false;

    vk::GraphicsPipelineCreateInfo lightingPipelineCreateInfo{};
    lightingPipelineCreateInfo.stageCount = 2;
    lightingPipelineCreateInfo.pStages = shaderStages;
    lightingPipelineCreateInfo.pVertexInputState = &vertexInputStateCreateInfo;
    lightingPipelineCreateInfo.pInputAssemblyState = &inputAssemblyStateCreateInfo;
    lightingPipelineCreateInfo.pViewportState = &viewportStateCreateInfo;
    lightingPipelineCreateInfo.pRasterizationState = &rasterizationStateCreateInfo;
    lightingPipelineCreateInfo.pMultisampleState = &multisampleStateCreateInfo;
    lightingPipelineCreateInfo.pDepthStencilState = &depthStencilStateCreateInfo;
    lightingPipelineCreateInfo.pColorBlendState = &colorBlendStateCreateInfo;
    lightingPipelineCreateInfo.pDynamicState = &dynamicStateCreateInfo;
    lightingPipelineCreateInfo.layout = _lightingPipelineLayout;
    lightingPipelineCreateInfo.subpass = 0;
    lightingPipelineCreateInfo.basePipelineHandle = nullptr;
    lightingPipelineCreateInfo.basePipelineIndex = -1;

    vk::PipelineRenderingCreateInfoKHR pipelineRenderingCreateInfoKhr{};
    pipelineRenderingCreateInfoKhr.colorAttachmentCount = 1;
    vk::Format format = _swapChain->GetFormat();
    pipelineRenderingCreateInfoKhr.pColorAttachmentFormats = &format;

    lightingPipelineCreateInfo.pNext = &pipelineRenderingCreateInfoKhr;
    lightingPipelineCreateInfo.renderPass = nullptr; // Using dynamic rendering.

    auto result = _device.createGraphicsPipeline(nullptr, lightingPipelineCreateInfo, nullptr);
    util::VK_ASSERT(result.result, "Failed creating the geometry pipeline layout!");
    _lightingPipeline = result.value;

    _device.destroy(vertModule);
    _device.destroy(fragModule);
}

void Engine::CreateCommandPool()
{
    vk::CommandPoolCreateInfo commandPoolCreateInfo{};
    commandPoolCreateInfo.flags = vk::CommandPoolCreateFlagBits::eResetCommandBuffer;
    commandPoolCreateInfo.queueFamilyIndex = _queueFamilyIndices.graphicsFamily.value();

    util::VK_ASSERT(_device.createCommandPool(&commandPoolCreateInfo, nullptr, &_commandPool), "Failed creating command pool!");
}

void Engine::CreateCommandBuffers()
{
    vk::CommandBufferAllocateInfo commandBufferAllocateInfo{};
    commandBufferAllocateInfo.commandPool = _commandPool;
    commandBufferAllocateInfo.level = vk::CommandBufferLevel::ePrimary;
    commandBufferAllocateInfo.commandBufferCount = _commandBuffers.size();

    util::VK_ASSERT(_device.allocateCommandBuffers(&commandBufferAllocateInfo, _commandBuffers.data()),
                    "Failed allocating command buffer!");
}

void Engine::RecordCommandBuffer(const vk::CommandBuffer &commandBuffer, uint32_t swapChainImageIndex)
{
    vk::CommandBufferBeginInfo commandBufferBeginInfo{};
    util::VK_ASSERT(commandBuffer.begin(&commandBufferBeginInfo), "Failed to begin recording command buffer!");

    util::TransitionImageLayout(commandBuffer, _swapChain->GetImage(swapChainImageIndex), _swapChain->GetFormat(), vk::ImageLayout::eUndefined, vk::ImageLayout::eColorAttachmentOptimal);

    for(auto& attachment : _frameData[_currentFrame].deferredAttachments)
        util::TransitionImageLayout(commandBuffer, attachment.image, vk::Format::eR16G16B16A16Sfloat, vk::ImageLayout::eUndefined, vk::ImageLayout::eColorAttachmentOptimal);

    std::array<vk::RenderingAttachmentInfoKHR, 3> colorAttachmentInfos{};
    vk::RenderingAttachmentInfoKHR& albedoAttachmentInfo{ colorAttachmentInfos[0] };
    albedoAttachmentInfo.imageView = _frameData[_currentFrame].deferredAttachments[0].imageView;
    albedoAttachmentInfo.imageLayout = vk::ImageLayout::eColorAttachmentOptimal;
    albedoAttachmentInfo.storeOp = vk::AttachmentStoreOp::eStore;
    albedoAttachmentInfo.loadOp = vk::AttachmentLoadOp::eClear;
    albedoAttachmentInfo.clearValue.color = vk::ClearColorValue{ 0.0f, 0.0f, 0.0f, 0.0f };

    vk::RenderingAttachmentInfoKHR& normalAttachmentInfo{ colorAttachmentInfos[1] };
    normalAttachmentInfo.imageView = _frameData[_currentFrame].deferredAttachments[1].imageView;
    normalAttachmentInfo.imageLayout = vk::ImageLayout::eColorAttachmentOptimal;
    normalAttachmentInfo.storeOp = vk::AttachmentStoreOp::eStore;
    normalAttachmentInfo.loadOp = vk::AttachmentLoadOp::eClear;
    normalAttachmentInfo.clearValue.color = vk::ClearColorValue{ 0.0f, 0.0f, 0.0f, 0.0f };

    vk::RenderingAttachmentInfoKHR& positionAttachmentInfo{ colorAttachmentInfos[2] };
    positionAttachmentInfo.imageView = _frameData[_currentFrame].deferredAttachments[2].imageView;
    positionAttachmentInfo.imageLayout = vk::ImageLayout::eColorAttachmentOptimal;
    positionAttachmentInfo.storeOp = vk::AttachmentStoreOp::eStore;
    positionAttachmentInfo.loadOp = vk::AttachmentLoadOp::eClear;
    positionAttachmentInfo.clearValue.color = vk::ClearColorValue{ 0.0f, 0.0f, 0.0f, 0.0f };

    vk::RenderingAttachmentInfoKHR depthAttachmentInfo{};
    depthAttachmentInfo.imageView = _swapChain->GetDepthView();
    depthAttachmentInfo.imageLayout = vk::ImageLayout::eDepthStencilAttachmentOptimal;
    depthAttachmentInfo.storeOp = vk::AttachmentStoreOp::eDontCare;
    depthAttachmentInfo.loadOp = vk::AttachmentLoadOp::eClear;
    depthAttachmentInfo.clearValue.depthStencil = vk::ClearDepthStencilValue{ 1.0f, 0 };

    vk::RenderingAttachmentInfoKHR  stencilAttachmentInfo{depthAttachmentInfo};
    stencilAttachmentInfo.storeOp = vk::AttachmentStoreOp::eDontCare;
    stencilAttachmentInfo.loadOp = vk::AttachmentLoadOp::eDontCare;
    stencilAttachmentInfo.clearValue.depthStencil = vk::ClearDepthStencilValue{ 1.0f, 0 };

    vk::RenderingInfoKHR renderingInfo{};
    glm::uvec2 displaySize = _swapChain->GetImageSize();
    renderingInfo.renderArea.extent = vk::Extent2D{ displaySize.x, displaySize.y };
    renderingInfo.renderArea.offset = vk::Offset2D{ 0, 0 };
    renderingInfo.colorAttachmentCount = colorAttachmentInfos.size();
    renderingInfo.pColorAttachments = colorAttachmentInfos.data();
    renderingInfo.layerCount = 1;
    renderingInfo.pDepthAttachment = &depthAttachmentInfo;
    renderingInfo.pStencilAttachment = util::HasStencilComponent(_swapChain->GetDepthFormat()) ? &stencilAttachmentInfo : nullptr;

    util::BeginLabel(commandBuffer, "Geometry pass", glm::vec3{ 1.0f, 0.0f, 0.0f }, _dldi);

    commandBuffer.beginRenderingKHR(&renderingInfo, _dldi);

    commandBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, _geometryPipeline);

    commandBuffer.setViewport(0, 1, &_viewport);
    commandBuffer.setScissor(0, 1, &_scissor);

    for(auto& mesh : _model.meshes)
    {
        for(auto& primitive : mesh.primitives)
        {
            if(primitive.topology != vk::PrimitiveTopology::eTriangleList)
                throw std::runtime_error("No support for topology other than triangle list!");

            if(!_model.textures.empty())
                UpdateGeometryDescriptorSet(_currentFrame, _model.textures[0].imageView);

            vk::Buffer vertexBuffers[] = { primitive.vertexBuffer };
            vk::DeviceSize offsets[] = { 0 };
            commandBuffer.bindVertexBuffers(0, 1, vertexBuffers, offsets);
            commandBuffer.bindIndexBuffer(primitive.indexBuffer, 0, primitive.indexType);

            commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, _geometryPipelineLayout, 0, 1, &_frameData[_currentFrame].geometryDescriptorSet, 0, nullptr);

            commandBuffer.drawIndexed(primitive.triangleCount, 1, 0, 0, 0);
        }
    }

    commandBuffer.endRenderingKHR(_dldi);

    util::EndLabel(commandBuffer, _dldi);


    for(auto& attachment : _frameData[_currentFrame].deferredAttachments)
        util::TransitionImageLayout(commandBuffer, attachment.image, vk::Format::eR16G16B16A16Sfloat, vk::ImageLayout::eColorAttachmentOptimal, vk::ImageLayout::eShaderReadOnlyOptimal);

    vk::RenderingAttachmentInfoKHR finalColorAttachmentInfo{};
    finalColorAttachmentInfo.imageView = _swapChain->GetImageView(swapChainImageIndex);
    finalColorAttachmentInfo.imageLayout = vk::ImageLayout::eAttachmentOptimalKHR;
    finalColorAttachmentInfo.storeOp = vk::AttachmentStoreOp::eStore;
    finalColorAttachmentInfo.loadOp = vk::AttachmentLoadOp::eClear;
    finalColorAttachmentInfo.clearValue.color = vk::ClearColorValue{ 0.0f, 0.0f, 0.0f, 0.0f };

    renderingInfo = vk::RenderingInfoKHR{};
    renderingInfo.renderArea.extent = vk::Extent2D{ displaySize.x, displaySize.y };
    renderingInfo.renderArea.offset = vk::Offset2D{ 0, 0 };
    renderingInfo.colorAttachmentCount = 1;
    renderingInfo.pColorAttachments = &finalColorAttachmentInfo;
    renderingInfo.layerCount = 1;
    renderingInfo.pDepthAttachment = nullptr;
    renderingInfo.pStencilAttachment = nullptr;

    util::BeginLabel(commandBuffer, "Lighting pass", glm::vec3{ 1.0f, 0.0f, 0.0f }, _dldi);
    commandBuffer.beginRenderingKHR(&renderingInfo, _dldi);

    commandBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, _lightingPipeline);

    commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, _lightingPipelineLayout, 0, 1, &_frameData[_currentFrame].lightingDescriptorSet, 0, nullptr);

    // Fullscreen quad.
    commandBuffer.draw(3, 1, 0, 0);

    ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), _commandBuffers[_currentFrame]);

    commandBuffer.endRenderingKHR(_dldi);
    util::EndLabel(commandBuffer, _dldi);

    util::TransitionImageLayout(commandBuffer, _swapChain->GetImage(swapChainImageIndex), _swapChain->GetFormat(), vk::ImageLayout::eColorAttachmentOptimal, vk::ImageLayout::ePresentSrcKHR);

    commandBuffer.end();
}

void Engine::CreateSyncObjects()
{
    vk::SemaphoreCreateInfo semaphoreCreateInfo{};
    vk::FenceCreateInfo fenceCreateInfo{};
    fenceCreateInfo.flags = vk::FenceCreateFlagBits::eSignaled;

    std::string errorMsg{ "Failed creating sync object!" };
    for(size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i)
    {
        util::VK_ASSERT(_device.createSemaphore(&semaphoreCreateInfo, nullptr, &_imageAvailableSemaphores[i]), errorMsg);
        util::VK_ASSERT(_device.createSemaphore(&semaphoreCreateInfo, nullptr, &_renderFinishedSemaphores[i]), errorMsg);
        util::VK_ASSERT(_device.createFence(&fenceCreateInfo, nullptr, &_inFlightFences[i]), errorMsg);
    }
}

void Engine::CreateBuffer(vk::DeviceSize size, vk::BufferUsageFlags usage, vk::Buffer& buffer, bool mappable, VmaAllocation& allocation) const
{
    vk::BufferCreateInfo bufferInfo{};
    bufferInfo.size = size;
    bufferInfo.usage = usage;
    bufferInfo.sharingMode = vk::SharingMode::eExclusive;
    bufferInfo.queueFamilyIndexCount = 1;
    bufferInfo.pQueueFamilyIndices = &_queueFamilyIndices.graphicsFamily.value();

    VmaAllocationCreateInfo allocationInfo{};
    allocationInfo.usage = VMA_MEMORY_USAGE_AUTO;
    if(mappable)
        allocationInfo.flags |= VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT;

    util::VK_ASSERT(vmaCreateBuffer(_vmaAllocator, reinterpret_cast<VkBufferCreateInfo*>(&bufferInfo), &allocationInfo, reinterpret_cast<VkBuffer*>(&buffer), &allocation, nullptr), "Failed creating buffer!");
}

template <typename T>
void Engine::CreateLocalBuffer(const std::vector<T>& vec, vk::Buffer& buffer, VmaAllocation& allocation, vk::BufferUsageFlags usage) const
{
    vk::DeviceSize bufferSize = vec.size() * sizeof(T);

    vk::Buffer stagingBuffer;
    VmaAllocation stagingBufferAllocation;
    CreateBuffer(bufferSize, vk::BufferUsageFlagBits::eTransferSrc, stagingBuffer, true, stagingBufferAllocation);

    vmaCopyMemoryToAllocation(_vmaAllocator, vec.data(), stagingBufferAllocation, 0, bufferSize);

    CreateBuffer(bufferSize, vk::BufferUsageFlagBits::eTransferDst | usage, buffer, false, allocation);

    CopyBuffer(stagingBuffer, buffer, bufferSize);
    _device.destroy(stagingBuffer, nullptr);
    vmaFreeMemory(_vmaAllocator, stagingBufferAllocation);
}

void Engine::CopyBuffer(vk::Buffer srcBuffer, vk::Buffer dstBuffer, vk::DeviceSize size) const
{
    vk::CommandBuffer commandBuffer = util::BeginSingleTimeCommands(_device, _commandPool);

    vk::BufferCopy copyRegion{};
    copyRegion.srcOffset = 0;
    copyRegion.dstOffset = 0;
    copyRegion.size = size;
    commandBuffer.copyBuffer(srcBuffer, dstBuffer, 1, &copyRegion);

    util::EndSingleTimeCommands(_device, _graphicsQueue, commandBuffer, _commandPool);
}

void Engine::CreateUniformBuffers()
{
    vk::DeviceSize bufferSize = sizeof(UBO);

    for(size_t i = 0; i < _frameData.size(); ++i)
    {
        CreateBuffer(bufferSize,
                     vk::BufferUsageFlagBits::eUniformBuffer,
                     _frameData[i].uniformBuffer, true, _frameData[i].uniformBufferAllocation);

        util::VK_ASSERT(vmaMapMemory(_vmaAllocator, _frameData[i].uniformBufferAllocation, &_frameData[i].uniformBufferMapped), "Failed mapping memory for UBO!");
    }
}

void Engine::UpdateUniformData(uint32_t currentFrame)
{
    static auto startTime = std::chrono::high_resolution_clock::now();

    auto currentTime = std::chrono::high_resolution_clock::now();
    float time = std::chrono::duration<float, std::chrono::seconds::period>(currentTime - startTime).count();

    UBO ubo{};
    ubo.model = glm::rotate(glm::mat4{1.0f}, time * glm::radians(90.0f), glm::vec3{0.0f, 1.0f, 0.0f});
    ubo.view = glm::lookAt(glm::vec3{3.0f}, glm::vec3{0.0f}, glm::vec3{0.0f, 1.0f, 0.0f});
    ubo.proj = glm::perspective(glm::radians(45.0f), _swapChain->GetExtent().width / static_cast<float>(_swapChain->GetExtent().height), 0.01f, 100.0f);
    ubo.proj[1][1] *= -1;

    memcpy(_frameData[currentFrame].uniformBufferMapped, &ubo, sizeof(ubo));
}

void Engine::CreateDescriptorSetLayout()
{
    // Geometry
    {
        std::array<vk::DescriptorSetLayoutBinding, 3> bindings{};

        vk::DescriptorSetLayoutBinding& descriptorSetLayoutBinding{bindings[0]};
        descriptorSetLayoutBinding.binding = 0;
        descriptorSetLayoutBinding.descriptorCount = 1;
        descriptorSetLayoutBinding.descriptorType = vk::DescriptorType::eUniformBuffer;
        descriptorSetLayoutBinding.stageFlags = vk::ShaderStageFlagBits::eVertex;
        descriptorSetLayoutBinding.pImmutableSamplers = nullptr;

        vk::DescriptorSetLayoutBinding& samplerLayoutBinding{bindings[1]};
        samplerLayoutBinding.binding = 1;
        samplerLayoutBinding.descriptorCount = 1;
        samplerLayoutBinding.descriptorType = vk::DescriptorType::eSampler;
        samplerLayoutBinding.stageFlags = vk::ShaderStageFlagBits::eFragment;
        samplerLayoutBinding.pImmutableSamplers = nullptr;

        vk::DescriptorSetLayoutBinding& imageLayoutBinding{bindings[2]};
        imageLayoutBinding.binding = 2;
        imageLayoutBinding.descriptorCount = 1;
        imageLayoutBinding.descriptorType = vk::DescriptorType::eSampledImage;
        imageLayoutBinding.stageFlags = vk::ShaderStageFlagBits::eFragment;
        imageLayoutBinding.pImmutableSamplers = nullptr;

        vk::DescriptorSetLayoutCreateInfo createInfo{};
        createInfo.bindingCount = bindings.size();
        createInfo.pBindings = bindings.data();

        util::VK_ASSERT(_device.createDescriptorSetLayout(&createInfo, nullptr, &_geometryDescriptorSetLayout),
                        "Failed creating geometry descriptor set layout!");
    }

    // Lighting
    {
        std::array<vk::DescriptorSetLayoutBinding, 4> bindings{};

        vk::DescriptorSetLayoutBinding& samplerLayoutBinding{bindings[0]};
        samplerLayoutBinding.binding = 0;
        samplerLayoutBinding.descriptorCount = 1;
        samplerLayoutBinding.descriptorType = vk::DescriptorType::eSampler;
        samplerLayoutBinding.stageFlags = vk::ShaderStageFlagBits::eFragment;
        samplerLayoutBinding.pImmutableSamplers = nullptr;

        vk::DescriptorSetLayoutBinding& albedoLayoutBinding{bindings[1]};
        albedoLayoutBinding.binding = 1;
        albedoLayoutBinding.descriptorCount = 1;
        albedoLayoutBinding.descriptorType = vk::DescriptorType::eSampledImage;
        albedoLayoutBinding.stageFlags = vk::ShaderStageFlagBits::eFragment;
        albedoLayoutBinding.pImmutableSamplers = nullptr;

        vk::DescriptorSetLayoutBinding& normalLayoutBinding{bindings[2]};
        normalLayoutBinding.binding = 2;
        normalLayoutBinding.descriptorCount = 1;
        normalLayoutBinding.descriptorType = vk::DescriptorType::eSampledImage;
        normalLayoutBinding.stageFlags = vk::ShaderStageFlagBits::eFragment;
        normalLayoutBinding.pImmutableSamplers = nullptr;

        vk::DescriptorSetLayoutBinding& positionLayoutBinding{bindings[3]};
        positionLayoutBinding.binding = 3;
        positionLayoutBinding.descriptorCount = 1;
        positionLayoutBinding.descriptorType = vk::DescriptorType::eSampledImage;
        positionLayoutBinding.stageFlags = vk::ShaderStageFlagBits::eFragment;
        positionLayoutBinding.pImmutableSamplers = nullptr;

        vk::DescriptorSetLayoutCreateInfo createInfo{};
        createInfo.bindingCount = bindings.size();
        createInfo.pBindings = bindings.data();

        util::VK_ASSERT(_device.createDescriptorSetLayout(&createInfo, nullptr, &_lightingDescriptorSetLayout),
                        "Failed creating lighting descriptor set layout!");
    }
}

void Engine::CreateDescriptorPool()
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
    createInfo.maxSets = MAX_FRAMES_IN_FLIGHT * 2 + 1;
    createInfo.flags = vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet;
    util::VK_ASSERT(_device.createDescriptorPool(&createInfo, nullptr, &_descriptorPool), "Failed creating descriptor pool!");
}

void Engine::CreateDescriptorSets()
{
    {
        std::array<vk::DescriptorSetLayout, MAX_FRAMES_IN_FLIGHT> layouts{};
        std::for_each(layouts.begin(), layouts.end(), [this](auto& l)
        { l = _geometryDescriptorSetLayout; });
        vk::DescriptorSetAllocateInfo allocateInfo{};
        allocateInfo.descriptorPool = _descriptorPool;
        allocateInfo.descriptorSetCount = MAX_FRAMES_IN_FLIGHT;
        allocateInfo.pSetLayouts = layouts.data();

        std::array<vk::DescriptorSet, MAX_FRAMES_IN_FLIGHT> descriptorSets;

        util::VK_ASSERT(_device.allocateDescriptorSets(&allocateInfo, descriptorSets.data()),
                        "Failed allocating descriptor sets!");
        for (size_t i = 0; i < descriptorSets.size(); ++i)
            _frameData[i].geometryDescriptorSet = descriptorSets[i];
    }
    {
        std::array<vk::DescriptorSetLayout, MAX_FRAMES_IN_FLIGHT> layouts{};
        std::for_each(layouts.begin(), layouts.end(), [this](auto& l) { l = _lightingDescriptorSetLayout; });
        vk::DescriptorSetAllocateInfo allocateInfo{};
        allocateInfo.descriptorPool = _descriptorPool;
        allocateInfo.descriptorSetCount = MAX_FRAMES_IN_FLIGHT;
        allocateInfo.pSetLayouts = layouts.data();

        std::array<vk::DescriptorSet, MAX_FRAMES_IN_FLIGHT> descriptorSets;

        util::VK_ASSERT(_device.allocateDescriptorSets(&allocateInfo, descriptorSets.data()),
                        "Failed allocating descriptor sets!");
        for (size_t i = 0; i < descriptorSets.size(); ++i)
            _frameData[i].lightingDescriptorSet = descriptorSets[i];
    }

    for(size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i)
    {
        if(!_model.textures.empty())
            UpdateGeometryDescriptorSet(i, _model.textures[0].imageView);

        UpdateLightingDescriptorSet(i);
    }
}

void Engine::CreateTextureImage(const Texture& texture, TextureHandle& textureHandle, vk::Format format)
{
    vk::DeviceSize imageSize = texture.width * texture.height * texture.numChannels;

    vk::Buffer stagingBuffer;
    VmaAllocation stagingBufferAllocation;

    CreateBuffer(imageSize, vk::BufferUsageFlagBits::eTransferSrc, stagingBuffer, true, stagingBufferAllocation);

    vmaCopyMemoryToAllocation(_vmaAllocator, texture.data.data(), stagingBufferAllocation, 0, imageSize);

    util::CreateImage(_device, _vmaAllocator, texture.width, texture.height, format,
                      vk::ImageTiling::eOptimal,
                      vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eSampled,
                      textureHandle.image, textureHandle.imageAllocation);

    vk::CommandBuffer commandBuffer = util::BeginSingleTimeCommands(_device, _commandPool);
    util::TransitionImageLayout(commandBuffer, textureHandle.image, format, vk::ImageLayout::eUndefined, vk::ImageLayout::eTransferDstOptimal);
    util::EndSingleTimeCommands(_device, _graphicsQueue, commandBuffer, _commandPool);

    CopyBufferToImage(stagingBuffer, textureHandle.image, texture.width, texture.height);

    commandBuffer = util::BeginSingleTimeCommands(_device, _commandPool);
    util::TransitionImageLayout(commandBuffer, textureHandle.image, format, vk::ImageLayout::eTransferDstOptimal, vk::ImageLayout::eShaderReadOnlyOptimal);
    util::EndSingleTimeCommands(_device, _graphicsQueue, commandBuffer, _commandPool);

    _device.destroy(stagingBuffer, nullptr);
    vmaFreeMemory(_vmaAllocator, stagingBufferAllocation);
}

void Engine::CopyBufferToImage(vk::Buffer buffer, vk::Image image, uint32_t width, uint32_t height)
{
    vk::BufferImageCopy region{};
    region.bufferImageHeight = 0;
    region.bufferRowLength = 0;
    region.bufferImageHeight = 0;
    region.imageSubresource.aspectMask = vk::ImageAspectFlagBits::eColor;
    region.imageSubresource.mipLevel = 0;
    region.imageSubresource.baseArrayLayer = 0;
    region.imageSubresource.layerCount = 1;
    region.imageOffset = vk::Offset3D{ 0, 0, 0 };
    region.imageExtent = vk::Extent3D{ width, height, 1 };

    vk::CommandBuffer commandBuffer = util::BeginSingleTimeCommands(_device, _commandPool);

    commandBuffer.copyBufferToImage(buffer, image, vk::ImageLayout::eTransferDstOptimal, 1, &region);

    util::EndSingleTimeCommands(_device, _graphicsQueue, commandBuffer, _commandPool);
}

void Engine::CreateTextureSampler()
{
    vk::PhysicalDeviceProperties properties{};
    _physicalDevice.getProperties(&properties);

    vk::SamplerCreateInfo createInfo{};
    createInfo.magFilter = vk::Filter::eLinear;
    createInfo.minFilter = vk::Filter::eLinear;
    createInfo.addressModeU = vk::SamplerAddressMode::eRepeat;
    createInfo.addressModeV = vk::SamplerAddressMode::eRepeat;
    createInfo.addressModeW = vk::SamplerAddressMode::eRepeat;
    createInfo.anisotropyEnable = vk::True;
    createInfo.maxAnisotropy = properties.limits.maxSamplerAnisotropy;
    createInfo.borderColor = vk::BorderColor::eIntOpaqueBlack;
    createInfo.unnormalizedCoordinates = vk::False;
    createInfo.compareEnable = vk::False;
    createInfo.compareOp = vk::CompareOp::eAlways;
    createInfo.mipmapMode = vk::SamplerMipmapMode::eLinear;
    createInfo.mipLodBias = 0.0f;
    createInfo.minLod = 0.0f;
    createInfo.maxLod = 0.0f;

    util::VK_ASSERT(_device.createSampler(&createInfo, nullptr, &_sampler), "Failed creating sampler!");
}

void Engine::UpdateGeometryDescriptorSet(uint32_t frameIndex, vk::ImageView texture)
{
    vk::DescriptorBufferInfo bufferInfo{};
    bufferInfo.buffer = _frameData[frameIndex].uniformBuffer;
    bufferInfo.offset = 0;
    bufferInfo.range = sizeof(UBO);

    vk::DescriptorImageInfo samplerInfo{};
    samplerInfo.sampler = _sampler;

    vk::DescriptorImageInfo imageInfo{};
    imageInfo.imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal;
    imageInfo.imageView = texture;

    std::array<vk::WriteDescriptorSet, 3> descriptorWrites{};

    vk::WriteDescriptorSet& bufferWrite{ descriptorWrites[0] };
    bufferWrite.dstSet = _frameData[frameIndex].geometryDescriptorSet;
    bufferWrite.dstBinding = 0;
    bufferWrite.dstArrayElement = 0;
    bufferWrite.descriptorType = vk::DescriptorType::eUniformBuffer;
    bufferWrite.descriptorCount = 1;
    bufferWrite.pBufferInfo = &bufferInfo;

    vk::WriteDescriptorSet& samplerWrite{ descriptorWrites[1] };
    samplerWrite.dstSet = _frameData[frameIndex].geometryDescriptorSet;
    samplerWrite.dstBinding = 1;
    samplerWrite.dstArrayElement = 0;
    samplerWrite.descriptorType = vk::DescriptorType::eSampler;
    samplerWrite.descriptorCount = 1;
    samplerWrite.pImageInfo = &samplerInfo;

    vk::WriteDescriptorSet& imageWrite{ descriptorWrites[2] };
    imageWrite.dstSet = _frameData[frameIndex].geometryDescriptorSet;
    imageWrite.dstBinding = 2;
    imageWrite.dstArrayElement = 0;
    imageWrite.descriptorType = vk::DescriptorType::eSampledImage;
    imageWrite.descriptorCount = 1;
    imageWrite.pImageInfo = &imageInfo;

    _device.updateDescriptorSets(descriptorWrites.size(), descriptorWrites.data(), 0, nullptr);
}

void Engine::UpdateLightingDescriptorSet(uint32_t frameIndex)
{
    vk::DescriptorImageInfo samplerInfo{};
    samplerInfo.sampler = _sampler;

    std::array<vk::DescriptorImageInfo, 3> imageInfo{};
    imageInfo[0].imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal;
    imageInfo[0].imageView = _frameData[frameIndex].deferredAttachments[0].imageView;
    imageInfo[1].imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal;
    imageInfo[1].imageView = _frameData[frameIndex].deferredAttachments[1].imageView;
    imageInfo[2].imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal;
    imageInfo[2].imageView = _frameData[frameIndex].deferredAttachments[2].imageView;

    std::array<vk::WriteDescriptorSet, 4> descriptorWrites{};

    descriptorWrites[0].dstSet = _frameData[frameIndex].lightingDescriptorSet;
    descriptorWrites[0].dstBinding = 0;
    descriptorWrites[0].dstArrayElement = 0;
    descriptorWrites[0].descriptorType = vk::DescriptorType::eSampler;
    descriptorWrites[0].descriptorCount = 1;
    descriptorWrites[0].pImageInfo = &samplerInfo;

    descriptorWrites[1].dstSet = _frameData[frameIndex].lightingDescriptorSet;
    descriptorWrites[1].dstBinding = 1;
    descriptorWrites[1].dstArrayElement = 0;
    descriptorWrites[1].descriptorType = vk::DescriptorType::eSampledImage;
    descriptorWrites[1].descriptorCount = 1;
    descriptorWrites[1].pImageInfo = &imageInfo[0];

    descriptorWrites[2].dstSet = _frameData[frameIndex].lightingDescriptorSet;
    descriptorWrites[2].dstBinding = 2;
    descriptorWrites[2].dstArrayElement = 0;
    descriptorWrites[2].descriptorType = vk::DescriptorType::eSampledImage;
    descriptorWrites[2].descriptorCount = 1;
    descriptorWrites[2].pImageInfo = &imageInfo[1];

    descriptorWrites[3].dstSet = _frameData[frameIndex].lightingDescriptorSet;
    descriptorWrites[3].dstBinding = 3;
    descriptorWrites[3].dstArrayElement = 0;
    descriptorWrites[3].descriptorType = vk::DescriptorType::eSampledImage;
    descriptorWrites[3].descriptorCount = 1;
    descriptorWrites[3].pImageInfo = &imageInfo[2];

    _device.updateDescriptorSets(descriptorWrites.size(), descriptorWrites.data(), 0, nullptr);
}


ModelHandle Engine::LoadModel(const Model& model)
{
    ModelHandle modelHandle{};

    for(const auto& mesh : model.meshes)
    {
        MeshHandle meshHandle{};

        for(const auto& primitive : mesh.primitives)
        {
            MeshPrimitiveHandle primitiveHandle{};
            primitiveHandle.topology = primitive.topology;
            primitiveHandle.indexType = primitive.indexType;
            primitiveHandle.triangleCount = primitive.indices.size() / (primitiveHandle.indexType == vk::IndexType::eUint16 ? 2 : 4);

            CreateLocalBuffer(primitive.vertices, primitiveHandle.vertexBuffer, primitiveHandle.vertexBufferAllocation, vk::BufferUsageFlagBits::eVertexBuffer);
            CreateLocalBuffer(primitive.indices, primitiveHandle.indexBuffer, primitiveHandle.indexBufferAllocation, vk::BufferUsageFlagBits::eIndexBuffer);

            meshHandle.primitives.emplace_back(primitiveHandle);
        }

        modelHandle.meshes.emplace_back(meshHandle);
    }

    for(const auto& texture : model.textures)
    {
        TextureHandle textureHandle{};
        textureHandle.format = texture.GetFormat();
        textureHandle.width = texture.width;
        textureHandle.height = texture.height;
        textureHandle.numChannels = texture.numChannels;

        CreateTextureImage(texture, textureHandle, textureHandle.format);
        textureHandle.imageView = util::CreateImageView(_device, textureHandle.image, texture.GetFormat(), vk::ImageAspectFlagBits::eColor);

        modelHandle.textures.emplace_back(textureHandle);
    }

    return modelHandle;
}

void Engine::InitializeDeferredRTs()
{
    vk::Format format = vk::Format::eR16G16B16A16Sfloat;
    for(size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i)
    {
        _frameData[i].deferredAttachments[0].name = "Albedo GBuffer";
        _frameData[i].deferredAttachments[1].name = "Normal GBuffer";
        _frameData[i].deferredAttachments[2].name = "Position GBuffer";

        for(auto& attachment : _frameData[i].deferredAttachments)
        {
            Texture texture;
            texture.width = _swapChain->GetImageSize().x;
            texture.height = _swapChain->GetImageSize().y;
            texture.numChannels = 4;
            util::CreateImage(_device, _vmaAllocator,
                              texture.width, texture.height,
                              format,
                              vk::ImageTiling::eOptimal, vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eSampled,
                              attachment.image, attachment.imageAllocation);
            attachment.imageView = util::CreateImageView(_device, attachment.image, format, vk::ImageAspectFlagBits::eColor);

            util::NameObject(attachment.image, "[IMAGE] " + attachment.name, _device, _dldi);
            util::NameObject(attachment.imageView, "[IMAGE VIEW] " + attachment.name, _device, _dldi);

            vk::CommandBuffer cb = util::BeginSingleTimeCommands(_device, _commandPool);
            util::TransitionImageLayout(cb, attachment.image, format, vk::ImageLayout::eUndefined, vk::ImageLayout::eColorAttachmentOptimal);
            util::EndSingleTimeCommands(_device, _graphicsQueue, cb, _commandPool);
        }
    }
}