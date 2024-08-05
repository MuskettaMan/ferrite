#include <iostream>
#include <cstdint>
#include <vector>
#include <algorithm>
#include <cstring>
#include <map>
#include <set>
#include <implot.h>
#include <iomanip>
#include <thread>
#define GLM_FORCE_RADIANS
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

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

    _swapChain = std::make_unique<SwapChain>(_device, _physicalDevice, _surface);

    QueueFamilyIndices familyIndices = FindQueueFamilies(_physicalDevice);
    _swapChain->CreateSwapChain(glm::uvec2{ initInfo.width, initInfo.height }, familyIndices);

    CreateRenderPass();
    CreateDescriptorSetLayout();
    CreateGraphicsPipeline();
    _swapChain->CreateFrameBuffers(_renderPass);
    CreateCommandPool();

    CreateVertexBuffer();
    CreateIndexBuffer();
    CreateUniformBuffers();

    CreateCommandBuffers();
    CreateSyncObjects();
    CreateDescriptorPool();
    CreateDescriptorSets();

    _newImGuiFrame = initInfo.newImGuiFrame;
    _shutdownImGui = initInfo.shutdownImGui;


    ImGui_ImplVulkan_InitInfo initInfoVulkan{};
    initInfoVulkan.RenderPass = _renderPass;
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
        _swapChain->RecreateSwapChain(_application->DisplaySize(), _renderPass, familyIndices);
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
        _swapChain->RecreateSwapChain(_application->DisplaySize(), _renderPass, _queueFamilyIndices);
    } else
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

    _device.destroy(_vertexBuffer);
    _device.freeMemory(_vertexBufferMemory);

    _device.destroy(_indexBuffer);
    _device.freeMemory(_indexBufferMemory);

    _device.destroy(_pipeline);
    _device.destroy(_pipelineLayout);
    _device.destroy(_renderPass);

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

    _device.getQueue(_queueFamilyIndices.graphicsFamily.value(), 0, &_graphicsQueue);
    _device.getQueue(_queueFamilyIndices.presentFamily.value(), 0, &_presentQueue);
    _device.getQueue(_queueFamilyIndices.tranferFamily.value(), 0, &_transferQueue);
}

void Engine::CreateDescriptorSetLayout()
{
    vk::DescriptorSetLayoutBinding descriptorSetLayoutBinding{};
    descriptorSetLayoutBinding.binding = 0;
    descriptorSetLayoutBinding.descriptorType = vk::DescriptorType::eUniformBuffer;
    descriptorSetLayoutBinding.descriptorCount = 1;
    descriptorSetLayoutBinding.stageFlags = vk::ShaderStageFlagBits::eVertex;
    descriptorSetLayoutBinding.pImmutableSamplers = nullptr;

    vk::DescriptorSetLayoutCreateInfo createInfo{};
    createInfo.bindingCount = 1;
    createInfo.pBindings = &descriptorSetLayoutBinding;

    util::VK_ASSERT(_device.createDescriptorSetLayout(&createInfo, nullptr, &_descriptorSetLayout), "Failed creating descriptor set layout!");
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
    graphicsPipelineCreateInfo.renderPass = _renderPass;
    graphicsPipelineCreateInfo.subpass = 0;
    graphicsPipelineCreateInfo.basePipelineHandle = nullptr;
    graphicsPipelineCreateInfo.basePipelineIndex = -1;

    auto result = _device.createGraphicsPipeline(nullptr, graphicsPipelineCreateInfo, nullptr);
    util::VK_ASSERT(result.result, "Failed creating the graphics pipeline layout!");
    _pipeline = result.value;

    _device.destroy(vertModule);
    _device.destroy(fragModule);
}

void Engine::CreateRenderPass()
{
    vk::AttachmentDescription colorAttachment{};
    colorAttachment.format = _swapChain->GetFormat();
    colorAttachment.samples = vk::SampleCountFlagBits::e1;
    colorAttachment.loadOp = vk::AttachmentLoadOp::eClear;
    colorAttachment.storeOp = vk::AttachmentStoreOp::eStore;
    colorAttachment.stencilLoadOp = vk::AttachmentLoadOp::eDontCare;
    colorAttachment.stencilStoreOp = vk::AttachmentStoreOp::eDontCare;
    colorAttachment.initialLayout = vk::ImageLayout::eUndefined;
    colorAttachment.finalLayout = vk::ImageLayout::ePresentSrcKHR;

    vk::AttachmentReference colorAttachmentReference{};
    colorAttachmentReference.attachment = 0;
    colorAttachmentReference.layout = vk::ImageLayout::eColorAttachmentOptimal;

    vk::SubpassDescription subpassDescription{};
    subpassDescription.pipelineBindPoint = vk::PipelineBindPoint::eGraphics;
    subpassDescription.colorAttachmentCount = 1;
    subpassDescription.pColorAttachments = &colorAttachmentReference;

    vk::SubpassDependency dependency{};
    dependency.srcSubpass = vk::SubpassExternal;
    dependency.dstSubpass = 0;
    dependency.srcStageMask = vk::PipelineStageFlagBits::eColorAttachmentOutput;
    dependency.dstStageMask = vk::PipelineStageFlagBits::eColorAttachmentOutput;
    dependency.srcAccessMask = vk::AccessFlagBits::eNone;
    dependency.dstAccessMask = vk::AccessFlagBits::eColorAttachmentWrite | vk::AccessFlagBits::eColorAttachmentRead;

    vk::RenderPassCreateInfo renderPassCreateInfo{};
    renderPassCreateInfo.attachmentCount = 1;
    renderPassCreateInfo.pAttachments = &colorAttachment;
    renderPassCreateInfo.subpassCount = 1;
    renderPassCreateInfo.pSubpasses = &subpassDescription;
    renderPassCreateInfo.dependencyCount = 1;
    renderPassCreateInfo.pDependencies = &dependency;

    util::VK_ASSERT(_device.createRenderPass(&renderPassCreateInfo, nullptr, &_renderPass), "Failed creating the render pass!");
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

    vk::RenderPassBeginInfo renderPassBeginInfo{};
    renderPassBeginInfo.renderPass = _renderPass;
    renderPassBeginInfo.framebuffer = _swapChain->GetFrameBuffer(swapChainImageIndex);
    renderPassBeginInfo.renderArea = vk::Rect2D{ vk::Offset2D{ 0, 0 }, _swapChain->GetExtent() };

    vk::ClearValue clearColor{ vk::ClearColorValue{ std::array<float, 4>{ 0.0f, 0.0f, 0.0f, 1.0f }}};
    renderPassBeginInfo.clearValueCount = 1;
    renderPassBeginInfo.pClearValues = &clearColor;

    commandBuffer.beginRenderPass(&renderPassBeginInfo, vk::SubpassContents::eInline);

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

    commandBuffer.endRenderPass();

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
    vk::CommandBufferAllocateInfo allocateInfo{};
    allocateInfo.level = vk::CommandBufferLevel::ePrimary;
    allocateInfo.commandPool = _transferCommandPool;
    allocateInfo.commandBufferCount = 1;

    vk::CommandBuffer commandBuffer;
    util::VK_ASSERT(_device.allocateCommandBuffers(&allocateInfo, &commandBuffer), "Failed allocating copy command buffer!");

    vk::CommandBufferBeginInfo beginInfo{};
    beginInfo.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit;

    util::VK_ASSERT(commandBuffer.begin(&beginInfo), "Failed beginning command buffer!");

    vk::BufferCopy copyRegion{};
    copyRegion.srcOffset = 0;
    copyRegion.dstOffset = 0;
    copyRegion.size = size;
    commandBuffer.copyBuffer(srcBuffer, dstBuffer, 1, &copyRegion);

    commandBuffer.end();

    vk::SubmitInfo submitInfo{};
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &commandBuffer;

    util::VK_ASSERT(_transferQueue.submit(1, &submitInfo, nullptr), "Failed submitting copy buffer to queue!");
    _transferQueue.waitIdle();

    _device.free(_transferCommandPool, commandBuffer);
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

    vk::DescriptorPoolCreateInfo createInfo{ vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet, 1000,
                                             static_cast<uint32_t>(poolSizes.size()), poolSizes.data() };
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

        vk::WriteDescriptorSet descriptorWrite{};
        descriptorWrite.dstSet = _frameData[i].descriptorSet;
        descriptorWrite.dstBinding = 0;
        descriptorWrite.dstArrayElement = 0;
        descriptorWrite.descriptorType = vk::DescriptorType::eUniformBuffer;
        descriptorWrite.descriptorCount = 1;
        descriptorWrite.pBufferInfo = &bufferInfo;

        _device.updateDescriptorSets(1, &descriptorWrite, 0, nullptr);
    }
}
