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
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

#include "imgui.h"
#include "engine.hpp"
#include "vulkan_validation.hpp"
#include "vulkan_helper.hpp"
#include "shaders/shader_loader.hpp"
#include "imgui_impl_vulkan.h"
#include "stopwatch.hpp"

Engine::Engine()
{
    ImGui::CreateContext();
    ImPlot::CreateContext();
}

void Engine::Init(const InitInfo &initInfo, std::shared_ptr<Application> application)
{
    _application = std::move(application);

    CreateInstance(initInfo);
    SetupDebugMessenger();

    _surface = initInfo.retrieveSurface(_instance);

    PickPhysicalDevice();
    CreateDevice();

    _dldi = vk::DispatchLoaderDynamic{ _instance, vkGetInstanceProcAddr, _device, vkGetDeviceProcAddr };

    _swapChain = std::make_unique<SwapChain>(_device, _physicalDevice, _surface);

    QueueFamilyIndices familyIndices = FindQueueFamilies(_physicalDevice);
    _swapChain->CreateSwapChain(glm::uvec2{ initInfo.width, initInfo.height }, familyIndices);

    CreateDescriptorSetLayout();
    CreateGraphicsPipeline();
    CreateCommandPool();

    CreateTextureImage();
    CreateTextureImageView();
    CreateTextureSampler();

    CreateVertexBuffer();
    CreateIndexBuffer();
    CreateUniformBuffers();

    CreateCommandBuffers();
    CreateSyncObjects();
    CreateDescriptorPool();
    CreateDescriptorSets();

    _newImGuiFrame = initInfo.newImGuiFrame;
    _shutdownImGui = initInfo.shutdownImGui;

    vk::PipelineRenderingCreateInfoKHR pipelineRenderingCreateInfoKhr{};
    pipelineRenderingCreateInfoKhr.colorAttachmentCount = 1;
    vk::Format format = _swapChain->GetFormat();
    pipelineRenderingCreateInfoKhr.pColorAttachmentFormats = &format;

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

    _device.resetFences(1, &_inFlightFences[_currentFrame]);

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

    util::VK_ASSERT(_graphicsQueue.submit(1, &submitInfo, _inFlightFences[_currentFrame]), "Failed submitting to graphics queue!");

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


    if(result == vk::Result::eErrorOutOfDateKHR || result == vk::Result::eSuboptimalKHR)
    {
        _swapChain->RecreateSwapChain(_application->DisplaySize(), _queueFamilyIndices);
    }
    else
    {
        util::VK_ASSERT(result, "Failed acquiring next image from swap chain!");
    }

    _device.waitIdle();

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
        util::DestroyDebugUtilsMessengerEXT(_instance, _debugMessenger, nullptr);

    _swapChain.reset();

    _device.destroy(_descriptorPool);

    _device.destroy(_commandPool);
    _device.destroy(_transferCommandPool);

    _device.destroy(_sampler);
    _device.destroy(_imageView);
    _device.destroy(_image);
    _device.freeMemory(_imageMemory);

    _device.destroy(_vertexBuffer);
    _device.freeMemory(_vertexBufferMemory);

    _device.destroy(_indexBuffer);
    _device.freeMemory(_indexBufferMemory);

    _device.destroy(_pipeline);
    _device.destroy(_pipelineLayout);

    for(size_t i = 0; i < _frameData.size(); ++i)
    {
        _device.destroy(_frameData[i].uniformBuffer);
        _device.freeMemory(_frameData[i].uniformBufferMemory);
    }

    _device.destroy(_descriptorSetLayout);

    _instance.destroy(_surface);
    _device.destroy();
    _instance.destroy();
}

void Engine::CreateInstance(const InitInfo &initInfo)
{
    CheckValidationLayerSupport();
    if(_enableValidationLayers && !CheckValidationLayerSupport())
        throw std::runtime_error("Validation layers requested, but not supported!");

    vk::ApplicationInfo appInfo{ "", vk::makeApiVersion(0, 0, 0, 0), "No engine", vk::makeApiVersion(0, 1, 0, 0),
                                 vk::makeApiVersion(0, 1, 0, 0) };

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
        createInfo.pNext = static_cast<vk::DebugUtilsMessengerCreateInfoEXT *>(&debugCreateInfo);
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

void Engine::LogInstanceExtensions()
{
    std::vector<vk::ExtensionProperties> extensions = vk::enumerateInstanceExtensionProperties();

    std::cout << "Available extensions: \n";
    for(const auto &extension: extensions)
        std::cout << '\t' << extension.extensionName << '\n';
}

std::vector<const char *> Engine::GetRequiredExtensions(const InitInfo &initInfo)
{
    std::vector<const char *> extensions(initInfo.extensions, initInfo.extensions + initInfo.extensionCount);
    if(_enableValidationLayers)
        extensions.emplace_back(vk::EXTDebugUtilsExtensionName);

    extensions.emplace_back(VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME);

    return extensions;
}

void Engine::SetupDebugMessenger()
{
    if(!_enableValidationLayers)
        return;

    vk::DebugUtilsMessengerCreateInfoEXT createInfo{};
    util::PopulateDebugMessengerCreateInfo(createInfo);
    createInfo.pUserData = nullptr;

    util::VK_ASSERT(util::CreateDebugUtilsMessengerEXT(_instance, &createInfo, nullptr, &_debugMessenger),
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

        if(queueFamilies[i].queueFlags & vk::QueueFlagBits::eTransfer && !(queueFamilies[i].queueFlags & vk::QueueFlagBits::eGraphics))
            indices.tranferFamily = i;

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

    if(!indices.tranferFamily.has_value())
        indices.tranferFamily = indices.graphicsFamily;

    return indices;
}

void Engine::CreateDevice()
{
    _queueFamilyIndices = FindQueueFamilies(_physicalDevice);
    vk::PhysicalDeviceFeatures deviceFeatures;
    _physicalDevice.getFeatures(&deviceFeatures);

    std::vector <vk::DeviceQueueCreateInfo> queueCreateInfos{};
    std::set <uint32_t> uniqueQueueFamilies = { _queueFamilyIndices.graphicsFamily.value(), _queueFamilyIndices.presentFamily.value(),
                                                _queueFamilyIndices.tranferFamily.value() };
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
    _device.getQueue(_queueFamilyIndices.tranferFamily.value(), 0, &_transferQueue);
}

void Engine::CreateGraphicsPipeline()
{
    auto vertByteCode = shader::ReadFile("shaders/triangle-v.spv");
    auto fragByteCode = shader::ReadFile("shaders/triangle-f.spv");

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

    std::vector<vk::DynamicState> dynamicStates = {
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
    rasterizationStateCreateInfo.cullMode = vk::CullModeFlagBits::eNone;
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
    colorBlendAttachmentState.srcColorBlendFactor = vk::BlendFactor::eOne;
    colorBlendAttachmentState.dstColorBlendFactor = vk::BlendFactor::eZero;
    colorBlendAttachmentState.colorBlendOp = vk::BlendOp::eAdd;
    colorBlendAttachmentState.srcAlphaBlendFactor = vk::BlendFactor::eOne;
    colorBlendAttachmentState.dstAlphaBlendFactor = vk::BlendFactor::eZero;
    colorBlendAttachmentState.alphaBlendOp = vk::BlendOp::eAdd;

    vk::PipelineColorBlendStateCreateInfo colorBlendStateCreateInfo{};
    colorBlendStateCreateInfo.logicOpEnable = vk::False;
    colorBlendStateCreateInfo.logicOp = vk::LogicOp::eCopy;
    colorBlendStateCreateInfo.attachmentCount = 1;
    colorBlendStateCreateInfo.pAttachments = &colorBlendAttachmentState;
    colorBlendStateCreateInfo.blendConstants = std::array<float, 4>{ 0.0f, 0.0f, 0.0f, 0.0f };

    vk::PipelineLayoutCreateInfo pipelineLayoutCreateInfo{};
    pipelineLayoutCreateInfo.setLayoutCount = 1;
    pipelineLayoutCreateInfo.pSetLayouts = &_descriptorSetLayout;
    pipelineLayoutCreateInfo.pushConstantRangeCount = 0;
    pipelineLayoutCreateInfo.pPushConstantRanges = nullptr;

    util::VK_ASSERT(_device.createPipelineLayout(&pipelineLayoutCreateInfo, nullptr, &_pipelineLayout),
                    "Failed creating a pipeline layout!");

    vk::GraphicsPipelineCreateInfo graphicsPipelineCreateInfo{};
    graphicsPipelineCreateInfo.stageCount = 2;
    graphicsPipelineCreateInfo.pStages = shaderStages;
    graphicsPipelineCreateInfo.pVertexInputState = &vertexInputStateCreateInfo;
    graphicsPipelineCreateInfo.pInputAssemblyState = &inputAssemblyStateCreateInfo;
    graphicsPipelineCreateInfo.pViewportState = &viewportStateCreateInfo;
    graphicsPipelineCreateInfo.pRasterizationState = &rasterizationStateCreateInfo;
    graphicsPipelineCreateInfo.pMultisampleState = &multisampleStateCreateInfo;
    graphicsPipelineCreateInfo.pDepthStencilState = nullptr;
    graphicsPipelineCreateInfo.pColorBlendState = &colorBlendStateCreateInfo;
    graphicsPipelineCreateInfo.pDynamicState = &dynamicStateCreateInfo;
    graphicsPipelineCreateInfo.layout = _pipelineLayout;
    graphicsPipelineCreateInfo.subpass = 0;
    graphicsPipelineCreateInfo.basePipelineHandle = nullptr;
    graphicsPipelineCreateInfo.basePipelineIndex = -1;

    vk::PipelineRenderingCreateInfoKHR pipelineRenderingCreateInfoKhr{};
    pipelineRenderingCreateInfoKhr.colorAttachmentCount = 1;
    vk::Format format = _swapChain->GetFormat();
    pipelineRenderingCreateInfoKhr.pColorAttachmentFormats = &format;

    graphicsPipelineCreateInfo.pNext = &pipelineRenderingCreateInfoKhr;
    graphicsPipelineCreateInfo.renderPass = nullptr; // Using dynamic rendering.

    auto result = _device.createGraphicsPipeline(nullptr, graphicsPipelineCreateInfo, nullptr);
    util::VK_ASSERT(result.result, "Failed creating the graphics pipeline layout!");
    _pipeline = result.value;

    _device.destroy(vertModule);
    _device.destroy(fragModule);
}

void Engine::CreateCommandPool()
{
    vk::CommandPoolCreateInfo commandPoolCreateInfo{};
    commandPoolCreateInfo.flags = vk::CommandPoolCreateFlagBits::eResetCommandBuffer;
    commandPoolCreateInfo.queueFamilyIndex = _queueFamilyIndices.graphicsFamily.value();

    util::VK_ASSERT(_device.createCommandPool(&commandPoolCreateInfo, nullptr, &_commandPool), "Failed creating command pool!");

    vk::CommandPoolCreateInfo transferCommandPoolCreateInfo{};
    transferCommandPoolCreateInfo.flags = vk::CommandPoolCreateFlagBits::eTransient;
    transferCommandPoolCreateInfo.queueFamilyIndex = _queueFamilyIndices.tranferFamily.value();

    util::VK_ASSERT(_device.createCommandPool(&transferCommandPoolCreateInfo, nullptr, &_transferCommandPool), "Failed creating transfer command pool!");
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

    vk::ImageMemoryBarrier swapchainBarrierToOptimal{};
    swapchainBarrierToOptimal.dstAccessMask = vk::AccessFlagBits::eColorAttachmentWrite;
    swapchainBarrierToOptimal.oldLayout = vk::ImageLayout::eUndefined;
    swapchainBarrierToOptimal.newLayout = vk::ImageLayout::eColorAttachmentOptimal;
    swapchainBarrierToOptimal.image = _swapChain->GetImage(swapChainImageIndex);
    swapchainBarrierToOptimal.subresourceRange.aspectMask = vk::ImageAspectFlagBits::eColor;
    swapchainBarrierToOptimal.subresourceRange.levelCount = 1;
    swapchainBarrierToOptimal.subresourceRange.baseMipLevel = 0;
    swapchainBarrierToOptimal.subresourceRange.layerCount = 1;
    swapchainBarrierToOptimal.subresourceRange.baseArrayLayer = 0;

    commandBuffer.pipelineBarrier(vk::PipelineStageFlagBits::eTopOfPipe, vk::PipelineStageFlagBits::eColorAttachmentOutput,
                                  vk::DependencyFlags{ 0 },
                                  0, nullptr,
                                  0, nullptr,
                                  1, &swapchainBarrierToOptimal);

    vk::RenderingAttachmentInfoKHR colorAttachmentInfo{};
    colorAttachmentInfo.imageView = _swapChain->GetImageView(swapChainImageIndex);
    colorAttachmentInfo.imageLayout = vk::ImageLayout::eAttachmentOptimalKHR;
    colorAttachmentInfo.storeOp = vk::AttachmentStoreOp::eStore;
    colorAttachmentInfo.loadOp = vk::AttachmentLoadOp::eClear;
    colorAttachmentInfo.clearValue = vk::ClearValue{ 0.0f };
    colorAttachmentInfo.resolveMode = vk::ResolveModeFlagBits::eNone;

    vk::RenderingInfoKHR renderingInfo{};
    glm::uvec2 displaySize = _application->DisplaySize();
    renderingInfo.renderArea.extent = vk::Extent2D{ displaySize.x, displaySize.y };
    renderingInfo.colorAttachmentCount = 1;
    renderingInfo.pColorAttachments = &colorAttachmentInfo;
    renderingInfo.layerCount = 1;
    renderingInfo.pDepthAttachment = nullptr;
    renderingInfo.pStencilAttachment = nullptr;

    commandBuffer.beginRenderingKHR(&renderingInfo, _dldi);

    commandBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, _pipeline);

    commandBuffer.setViewport(0, 1, &_viewport);
    commandBuffer.setScissor(0, 1, &_scissor);

    vk::Buffer vertexBuffers[] = { _vertexBuffer };
    vk::DeviceSize offsets[] = { 0 };
    commandBuffer.bindVertexBuffers(0, 1, vertexBuffers, offsets);
    commandBuffer.bindIndexBuffer(_indexBuffer, 0, vk::IndexType::eUint16);

    commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, _pipelineLayout, 0, 1, &_frameData[_currentFrame].descriptorSet, 0, nullptr);

    commandBuffer.drawIndexed(_indices.size(), 1, 0, 0, 0);

    ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), _commandBuffers[_currentFrame]);

    commandBuffer.endRenderingKHR(_dldi);

    vk::ImageMemoryBarrier swapchainBarrierToPresent{};
    swapchainBarrierToPresent.srcAccessMask = vk::AccessFlagBits::eColorAttachmentWrite;
    swapchainBarrierToPresent.oldLayout = vk::ImageLayout::eColorAttachmentOptimal;
    swapchainBarrierToPresent.newLayout = vk::ImageLayout::ePresentSrcKHR;
    swapchainBarrierToPresent.image = _swapChain->GetImage(swapChainImageIndex);
    swapchainBarrierToPresent.subresourceRange.aspectMask = vk::ImageAspectFlagBits::eColor;
    swapchainBarrierToPresent.subresourceRange.levelCount = 1;
    swapchainBarrierToPresent.subresourceRange.baseMipLevel = 0;
    swapchainBarrierToPresent.subresourceRange.layerCount = 1;
    swapchainBarrierToPresent.subresourceRange.baseArrayLayer = 0;

    commandBuffer.pipelineBarrier(vk::PipelineStageFlagBits::eColorAttachmentOutput, vk::PipelineStageFlagBits::eBottomOfPipe,
                                  vk::DependencyFlags{ 0 },
                                  0, nullptr,
                                  0, nullptr,
                                  1, &swapchainBarrierToPresent);

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

void Engine::CreateBuffer(vk::DeviceSize size, vk::BufferUsageFlags usage, vk::MemoryPropertyFlags properties, vk::Buffer &buffer,
                          vk::DeviceMemory &bufferMemory)
{
    vk::BufferCreateInfo bufferInfo{};
    bufferInfo.size = size;
    bufferInfo.usage = usage;
    bufferInfo.sharingMode = vk::SharingMode::eExclusive;
    bufferInfo.queueFamilyIndexCount = 2;
    std::array<uint32_t, 2> concurrentFamilies = { _queueFamilyIndices.graphicsFamily.value(), _queueFamilyIndices.tranferFamily.value() };
    bufferInfo.pQueueFamilyIndices = concurrentFamilies.data();

    util::VK_ASSERT(_device.createBuffer(&bufferInfo, nullptr, &buffer), "Failed creating vertex buffer!");

    vk::MemoryRequirements memoryRequirements;
    _device.getBufferMemoryRequirements(buffer, &memoryRequirements);

    vk::MemoryAllocateInfo allocateInfo{};
    allocateInfo.allocationSize = memoryRequirements.size;
    allocateInfo.memoryTypeIndex = FindMemoryType(memoryRequirements.memoryTypeBits, properties);

    util::VK_ASSERT(_device.allocateMemory(&allocateInfo, nullptr, &bufferMemory), "Failed allocating memory for the vertex buffer!");

    _device.bindBufferMemory(buffer, bufferMemory, 0);
}

void Engine::CreateVertexBuffer()
{
    vk::DeviceSize bufferSize = sizeof(Vertex) * _vertices.size();

    vk::Buffer stagingBuffer;
    vk::DeviceMemory stagingBufferMemory;
    CreateBuffer(bufferSize, vk::BufferUsageFlagBits::eTransferSrc,
                 vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent, stagingBuffer, stagingBufferMemory);

    void *data;
    _device.mapMemory(stagingBufferMemory, 0, bufferSize, vk::MemoryMapFlags{ 0 }, &data);
    memcpy(data, _vertices.data(), static_cast<size_t>(bufferSize));
    _device.unmapMemory(stagingBufferMemory);

    CreateBuffer(bufferSize, vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eVertexBuffer,
                 vk::MemoryPropertyFlagBits::eDeviceLocal, _vertexBuffer, _vertexBufferMemory);

    CopyBuffer(stagingBuffer, _vertexBuffer, bufferSize);

    _device.destroy(stagingBuffer, nullptr);
    _device.freeMemory(stagingBufferMemory, nullptr);
}

void Engine::CreateIndexBuffer()
{
    vk::DeviceSize bufferSize = sizeof(_indices[0]) * _indices.size();


    vk::Buffer stagingBuffer;
    vk::DeviceMemory stagingBufferMemory;
    CreateBuffer(bufferSize,
                 vk::BufferUsageFlagBits::eTransferSrc,
                 vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
                 stagingBuffer, stagingBufferMemory);

    void* data;
    _device.mapMemory(stagingBufferMemory, 0, bufferSize, vk::MemoryMapFlags{ 0 }, &data);
    memcpy(data, _indices.data(), static_cast<size_t>(bufferSize));
    _device.unmapMemory(stagingBufferMemory);

    CreateBuffer(bufferSize,
                 vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eIndexBuffer,
                 vk::MemoryPropertyFlagBits::eDeviceLocal,
                 _indexBuffer, _indexBufferMemory);

    CopyBuffer(stagingBuffer, _indexBuffer, bufferSize);

    _device.destroy(stagingBuffer, nullptr);
    _device.freeMemory(stagingBufferMemory, nullptr);
}

void Engine::CopyBuffer(vk::Buffer srcBuffer, vk::Buffer dstBuffer, vk::DeviceSize size)
{
    vk::CommandBuffer commandBuffer = BeginSingleTimeCommands(_transferCommandPool);

    vk::BufferCopy copyRegion{};
    copyRegion.srcOffset = 0;
    copyRegion.dstOffset = 0;
    copyRegion.size = size;
    commandBuffer.copyBuffer(srcBuffer, dstBuffer, 1, &copyRegion);

    EndSingleTimeCommands(commandBuffer, _transferQueue, _transferCommandPool);
}

void Engine::CreateUniformBuffers()
{
    vk::DeviceSize bufferSize = sizeof(UBO);

    for(size_t i = 0; i < _frameData.size(); ++i)
    {
        CreateBuffer(bufferSize,
                     vk::BufferUsageFlagBits::eUniformBuffer,
                     vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
                     _frameData[i].uniformBuffer, _frameData[i].uniformBufferMemory);
        util::VK_ASSERT(_device.mapMemory(_frameData[i].uniformBufferMemory, vk::DeviceSize{ 0 }, bufferSize, vk::MemoryMapFlags{ 0 }, &_frameData[i].uniformBufferMapped), "Failed mapping memory for UBO!");
    }
}

void Engine::UpdateUniformData(uint32_t currentFrame)
{
    static auto startTime = std::chrono::high_resolution_clock::now();

    auto currentTime = std::chrono::high_resolution_clock::now();
    float time = std::chrono::duration<float, std::chrono::seconds::period>(currentTime - startTime).count();

    UBO ubo{};
    ubo.model = glm::rotate(glm::mat4{1.0f}, time * glm::radians(90.0f), glm::vec3{0.0f, 1.0f, 0.0f});
    ubo.view = glm::lookAt(glm::vec3{2.0f}, glm::vec3{0.0f}, glm::vec3{0.0f, 1.0f, 0.0f});
    ubo.proj = glm::perspective(glm::radians(45.0f), _swapChain->GetExtent().width / static_cast<float>(_swapChain->GetExtent().height), 0.1f, 100.0f);
    ubo.proj[1][1] *= -1;

    memcpy(_frameData[currentFrame].uniformBufferMapped, &ubo, sizeof(ubo));
}

uint32_t Engine::FindMemoryType(uint32_t typeFilter, vk::MemoryPropertyFlags properties)
{
    vk::PhysicalDeviceMemoryProperties memoryProperties;
    _physicalDevice.getMemoryProperties(&memoryProperties);

    for(uint32_t i = 0; i < memoryProperties.memoryTypeCount; ++i)
        if(typeFilter & (1 << i) && (memoryProperties.memoryTypes[i].propertyFlags & properties) == properties)
            return i;

    throw std::runtime_error("Failed finding suitable memory type!");
}

void Engine::CreateDescriptorSetLayout()
{
    std::array<vk::DescriptorSetLayoutBinding, 2> bindings{};

    vk::DescriptorSetLayoutBinding& descriptorSetLayoutBinding{ bindings[0] };
    descriptorSetLayoutBinding.binding = 0;
    descriptorSetLayoutBinding.descriptorCount = 1;
    descriptorSetLayoutBinding.descriptorType = vk::DescriptorType::eUniformBuffer;
    descriptorSetLayoutBinding.stageFlags = vk::ShaderStageFlagBits::eVertex;
    descriptorSetLayoutBinding.pImmutableSamplers = nullptr;

    vk::DescriptorSetLayoutBinding& samplerLayoutBinding{ bindings[1] };
    samplerLayoutBinding.binding = 1;
    samplerLayoutBinding.descriptorCount = 1;
    samplerLayoutBinding.descriptorType = vk::DescriptorType::eCombinedImageSampler;
    samplerLayoutBinding.stageFlags = vk::ShaderStageFlagBits::eFragment;
    samplerLayoutBinding.pImmutableSamplers = nullptr;


    vk::DescriptorSetLayoutCreateInfo createInfo{};
    createInfo.bindingCount = bindings.size();
    createInfo.pBindings = bindings.data();

    util::VK_ASSERT(_device.createDescriptorSetLayout(&createInfo, nullptr, &_descriptorSetLayout), "Failed creating descriptor set layout!");
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
    createInfo.maxSets = MAX_FRAMES_IN_FLIGHT + 1;
    createInfo.flags = vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet;
    util::VK_ASSERT(_device.createDescriptorPool(&createInfo, nullptr, &_descriptorPool), "Failed creating descriptor pool!");
}

void Engine::CreateDescriptorSets()
{
    std::array<vk::DescriptorSetLayout, MAX_FRAMES_IN_FLIGHT> layouts{};
    std::for_each(layouts.begin(), layouts.end(), [this](auto& l){ l = _descriptorSetLayout; });
    vk::DescriptorSetAllocateInfo allocateInfo{};
    allocateInfo.descriptorPool = _descriptorPool;
    allocateInfo.descriptorSetCount = MAX_FRAMES_IN_FLIGHT;
    allocateInfo.pSetLayouts = layouts.data();

    std::array<vk::DescriptorSet,  MAX_FRAMES_IN_FLIGHT> descriptorSets;

    util::VK_ASSERT(_device.allocateDescriptorSets(&allocateInfo, descriptorSets.data()), "Failed allocating descriptor sets!");
    for(size_t i = 0; i < descriptorSets.size(); ++i)
        _frameData[i].descriptorSet = descriptorSets[i];

    for(size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i)
    {
        vk::DescriptorBufferInfo bufferInfo{};
        bufferInfo.buffer = _frameData[i].uniformBuffer;
        bufferInfo.offset = 0;
        bufferInfo.range = sizeof(UBO);

        vk::DescriptorImageInfo imageInfo{};
        imageInfo.imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal;
        imageInfo.imageView = _imageView;
        imageInfo.sampler = _sampler;

        std::array<vk::WriteDescriptorSet, 2> descriptorWrites{};

        vk::WriteDescriptorSet& bufferWrite{ descriptorWrites[0] };
        bufferWrite.dstSet = _frameData[i].descriptorSet;
        bufferWrite.dstBinding = 0;
        bufferWrite.dstArrayElement = 0;
        bufferWrite.descriptorType = vk::DescriptorType::eUniformBuffer;
        bufferWrite.descriptorCount = 1;
        bufferWrite.pBufferInfo = &bufferInfo;

        vk::WriteDescriptorSet& imageWrite{ descriptorWrites[1] };
        imageWrite.dstSet = _frameData[i].descriptorSet;
        imageWrite.dstBinding = 1;
        imageWrite.dstArrayElement = 0;
        imageWrite.descriptorType = vk::DescriptorType::eCombinedImageSampler;
        imageWrite.descriptorCount = 1;
        imageWrite.pImageInfo = &imageInfo;

        _device.updateDescriptorSets(descriptorWrites.size(), descriptorWrites.data(), 0, nullptr);
    }
}

void Engine::CreateTextureImage()
{
    int32_t width, height, channels;

    stbi_set_flip_vertically_on_load(true);
    stbi_uc* pixels = stbi_load("assets/textures/kitty.jpg", &width, &height, &channels, STBI_rgb_alpha);
    vk::DeviceSize imageSize = width * height * 4;

    if(!pixels)
        throw std::runtime_error("Failed to load texture!");

    vk::Buffer stagingBuffer;
    vk::DeviceMemory stagingBufferMemory;

    CreateBuffer(imageSize, vk::BufferUsageFlagBits::eTransferSrc, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent, stagingBuffer, stagingBufferMemory);

    void* data;
    util::VK_ASSERT(_device.mapMemory(stagingBufferMemory, 0, imageSize, vk::MemoryMapFlags{0}, &data), "Failed mapping image memory!");
    memcpy(data, pixels, static_cast<size_t>(imageSize));
    _device.unmapMemory(stagingBufferMemory);

    stbi_image_free(pixels);

    CreateImage(width, height,
                vk::Format::eR8G8B8A8Srgb,
                vk::ImageTiling::eOptimal,
                vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eSampled,
                vk::MemoryPropertyFlagBits::eDeviceLocal,
                _image, _imageMemory);

    TransitionImageLayout(_image, vk::Format::eR8G8B8A8Srgb, vk::ImageLayout::eUndefined, vk::ImageLayout::eTransferDstOptimal);
    CopyBufferToImage(stagingBuffer, _image, width, height);
    TransitionImageLayout(_image, vk::Format::eR8G8B8A8Srgb, vk::ImageLayout::eTransferDstOptimal, vk::ImageLayout::eShaderReadOnlyOptimal);

    _device.destroy(stagingBuffer, nullptr);
    _device.free(stagingBufferMemory, nullptr);
}

void Engine::CreateImage(uint32_t width, uint32_t height, vk::Format format, vk::ImageTiling tiling, vk::ImageUsageFlags usage, vk::MemoryPropertyFlags properties, vk::Image& image, vk::DeviceMemory& memory)
{
    vk::ImageCreateInfo createInfo{};
    createInfo.imageType = vk::ImageType::e2D;
    createInfo.extent.width = width;
    createInfo.extent.height = height;
    createInfo.extent.depth = 1;
    createInfo.mipLevels = 1;
    createInfo.arrayLayers = 1;
    createInfo.format = format;
    createInfo.tiling = tiling;
    createInfo.initialLayout = vk::ImageLayout::eUndefined;
    createInfo.usage = usage;
    createInfo.sharingMode = vk::SharingMode::eExclusive;
    createInfo.samples = vk::SampleCountFlagBits::e1;
    createInfo.flags = vk::ImageCreateFlags{ 0 };

    util::VK_ASSERT(_device.createImage(&createInfo, nullptr, &image), "Failed creating image!");

    vk::MemoryRequirements memoryRequirements;
    _device.getImageMemoryRequirements(image, &memoryRequirements);

    vk::MemoryAllocateInfo allocateInfo{};
    allocateInfo.allocationSize = memoryRequirements.size;
    allocateInfo.memoryTypeIndex = FindMemoryType(memoryRequirements.memoryTypeBits, properties);

    util::VK_ASSERT(_device.allocateMemory(&allocateInfo, nullptr, &memory), "Failed allocating memory!");

    _device.bindImageMemory(image, memory, 0);
}

vk::CommandBuffer Engine::BeginSingleTimeCommands(vk::CommandPool& commandPool)
{
    vk::CommandBufferAllocateInfo allocateInfo{};
    allocateInfo.level = vk::CommandBufferLevel::ePrimary;
    allocateInfo.commandPool = commandPool;
    allocateInfo.commandBufferCount = 1;

    vk::CommandBuffer commandBuffer;
    util::VK_ASSERT(_device.allocateCommandBuffers(&allocateInfo, &commandBuffer), "Failed allocating one time command buffer!");

    vk::CommandBufferBeginInfo beginInfo{};
    beginInfo.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit;

    util::VK_ASSERT(commandBuffer.begin(&beginInfo), "Failed beginning one time command buffer!");

    return commandBuffer;
}

void Engine::EndSingleTimeCommands(vk::CommandBuffer commandBuffer, vk::Queue& queue, vk::CommandPool& commandPool)
{
    commandBuffer.end();

    vk::SubmitInfo submitInfo{};
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &commandBuffer;

    util::VK_ASSERT(queue.submit(1, &submitInfo, nullptr), "Failed submitting one time buffer to queue!");
    queue.waitIdle();

    _device.free(commandPool, commandBuffer);
}

void Engine::TransitionImageLayout(vk::Image image, vk::Format format, vk::ImageLayout oldLayout, vk::ImageLayout newLayout)
{
    vk::CommandBuffer commandBuffer = BeginSingleTimeCommands(_commandPool);

    vk::ImageMemoryBarrier barrier{};
    barrier.oldLayout = oldLayout;
    barrier.newLayout = newLayout;
    barrier.srcQueueFamilyIndex = vk::QueueFamilyIgnored;
    barrier.dstQueueFamilyIndex = vk::QueueFamilyIgnored;
    barrier.image = image;
    barrier.subresourceRange.aspectMask = vk::ImageAspectFlagBits::eColor;
    barrier.subresourceRange.baseMipLevel = 0;
    barrier.subresourceRange.levelCount = 1;
    barrier.subresourceRange.baseArrayLayer = 0;
    barrier.subresourceRange.layerCount = 1;

    vk::PipelineStageFlags sourceStage;
    vk::PipelineStageFlags destinationStage;

    if(oldLayout == vk::ImageLayout::eUndefined && newLayout == vk::ImageLayout::eTransferDstOptimal)
    {
        barrier.srcAccessMask = vk::AccessFlags{ 0 };
        barrier.dstAccessMask = vk::AccessFlagBits::eTransferWrite;

        sourceStage = vk::PipelineStageFlagBits::eTopOfPipe;
        destinationStage = vk::PipelineStageFlagBits::eTransfer;
    }
    else if (oldLayout == vk::ImageLayout::eTransferDstOptimal && newLayout == vk::ImageLayout::eShaderReadOnlyOptimal)
    {
        barrier.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
        barrier.dstAccessMask = vk::AccessFlagBits::eShaderRead;


        sourceStage = vk::PipelineStageFlagBits::eTransfer;
        destinationStage = vk::PipelineStageFlagBits::eFragmentShader;
    }
    else
        throw std::runtime_error("Unsupported layout transition!");

    commandBuffer.pipelineBarrier(sourceStage, destinationStage,
                                  vk::DependencyFlags{ 0 },
                                  0, nullptr,
                                  0, nullptr,
                                  1, &barrier);

    EndSingleTimeCommands(commandBuffer, _graphicsQueue, _commandPool);
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

    vk::CommandBuffer commandBuffer = BeginSingleTimeCommands(_commandPool);

    commandBuffer.copyBufferToImage(buffer, image, vk::ImageLayout::eTransferDstOptimal, 1, &region);

    EndSingleTimeCommands(commandBuffer, _graphicsQueue, _commandPool);
}

void Engine::CreateTextureImageView()
{
    _imageView = CreateImageView(_image, vk::Format::eR8G8B8A8Srgb);
}

vk::ImageView Engine::CreateImageView(vk::Image image, vk::Format format)
{
    vk::ImageViewCreateInfo createInfo{};
    createInfo.image = image;
    createInfo.viewType = vk::ImageViewType::e2D;
    createInfo.format = format;
    createInfo.subresourceRange.aspectMask = vk::ImageAspectFlagBits::eColor;
    createInfo.subresourceRange.baseMipLevel = 0;
    createInfo.subresourceRange.levelCount = 1;
    createInfo.subresourceRange.baseArrayLayer = 0;
    createInfo.subresourceRange.layerCount = 1;

    vk::ImageView view;
    util::VK_ASSERT(_device.createImageView(&createInfo, nullptr, &view), "Failed creating image view!");

    return view;
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

