#include <iostream>
#include <cstdint>
#include <vector>
#include <algorithm>
#include <cstring>
#include <set>
#include <implot.h>
#include <thread>
#include "spdlog/spdlog.h"
#include <spdlog/fmt/bundled/printf.h>

#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

#define VMA_IMPLEMENTATION
#define VMA_LEAK_LOG_FORMAT(format, ...) do { \
        spdlog::info(fmt::sprintf(format, __VA_ARGS__)); \
    } while(false)
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
#include "vulkan_brain.hpp"

Engine::Engine(const InitInfo& initInfo, std::shared_ptr<Application> application) :
    _brain(initInfo)
{
    ImGui::CreateContext();
    ImPlot::CreateContext();
    spdlog::info("Starting engine...");

    _application = std::move(application);


    auto supportedDepthFormat = util::FindSupportedFormat(_brain.physicalDevice, { vk::Format::eD32Sfloat, vk::Format::eD32SfloatS8Uint, vk::Format::eD24UnormS8Uint },
                                                          vk::ImageTiling::eOptimal,
                                                          vk::FormatFeatureFlagBits::eDepthStencilAttachment);

    assert(supportedDepthFormat.has_value() && "No supported depth format!");

    _swapChain = std::make_unique<SwapChain>(_brain, supportedDepthFormat.value());

    _swapChain->CreateSwapChain(glm::uvec2{ initInfo.width, initInfo.height }, _brain.queueFamilyIndices);


    CreateDescriptorSetLayout();
    InitializeDeferredRTs();

    CreateGeometryPipeline();
    CreateLightingPipeline();

    CreateTextureSampler();

    CreateUniformBuffers();

    CreateCommandBuffers();
    CreateSyncObjects();

    CreateDefaultMaterial();


    ModelLoader modelLoader;
    Model model = modelLoader.Load("assets/models/ABeautifulGame/ABeautifulGame.gltf");
    //Model model = modelLoader.Load("assets/models/DamagedHelmet.glb");
    _model = LoadModel(model);

    CreateDescriptorSets();

    vk::Format format = _swapChain->GetFormat();
    vk::PipelineRenderingCreateInfoKHR pipelineRenderingCreateInfoKhr{};
    pipelineRenderingCreateInfoKhr.colorAttachmentCount = 1;
    pipelineRenderingCreateInfoKhr.pColorAttachmentFormats = &format;
    pipelineRenderingCreateInfoKhr.depthAttachmentFormat = _swapChain->GetDepthFormat();

    _application->InitImGui();

    ImGui_ImplVulkan_InitInfo initInfoVulkan{};
    initInfoVulkan.UseDynamicRendering = true;
    initInfoVulkan.PipelineRenderingCreateInfo = static_cast<VkPipelineRenderingCreateInfo>(pipelineRenderingCreateInfoKhr);
    initInfoVulkan.PhysicalDevice = _brain.physicalDevice;
    initInfoVulkan.Device = _brain.device;
    initInfoVulkan.ImageCount = MAX_FRAMES_IN_FLIGHT;
    initInfoVulkan.Instance = _brain.instance;
    initInfoVulkan.MSAASamples = VkSampleCountFlagBits::VK_SAMPLE_COUNT_1_BIT;
    initInfoVulkan.Queue = _brain.graphicsQueue;
    initInfoVulkan.QueueFamily = _brain.queueFamilyIndices.graphicsFamily.value();
    initInfoVulkan.DescriptorPool = _brain.descriptorPool;
    initInfoVulkan.MinImageCount = 2;
    initInfoVulkan.ImageCount = _swapChain->GetImageCount();
    ImGui_ImplVulkan_Init(&initInfoVulkan);

    ImGui_ImplVulkan_CreateFontsTexture();

    spdlog::info("Successfully initialized engine!");
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

    util::VK_ASSERT(_brain.device.waitForFences(1, &_inFlightFences[_currentFrame], vk::True, std::numeric_limits<uint64_t>::max()),
                    "Failed waiting on in flight fence!");

    uint32_t imageIndex;
    vk::Result result = _brain.device.acquireNextImageKHR(_swapChain->GetSwapChain(), std::numeric_limits<uint64_t>::max(),
                                                    _imageAvailableSemaphores[_currentFrame], nullptr, &imageIndex);

    if(result == vk::Result::eErrorOutOfDateKHR || result == vk::Result::eSuboptimalKHR)
    {
        _swapChain->RecreateSwapChain(_application->DisplaySize(), _brain.queueFamilyIndices);

        return;
    } else
        util::VK_ASSERT(result, "Failed acquiring next image from swap chain!");

    util::VK_ASSERT(_brain.device.resetFences(1, &_inFlightFences[_currentFrame]), "Failed resetting fences!");

    UpdateUniformData(_currentFrame);

    ImGui_ImplVulkan_NewFrame();
    _application->NewImGuiFrame();
    ImGui::NewFrame();

    _performanceTracker.Render();

    ImGui::Render();

    _commandBuffers[_currentFrame].reset();

    RecordCommandBuffer(_commandBuffers[_currentFrame], imageIndex);

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

    util::VK_ASSERT(_brain.graphicsQueue.submit(1, &submitInfo, _inFlightFences[_currentFrame]), "Failed submitting to graphics queue!");

    vk::PresentInfoKHR presentInfo{};
    presentInfo.waitSemaphoreCount = 1;
    presentInfo.pWaitSemaphores = signalSemaphores;

    vk::SwapchainKHR swapchains[] = { _swapChain->GetSwapChain() };
    presentInfo.swapchainCount = 1;
    presentInfo.pSwapchains = swapchains;
    presentInfo.pImageIndices = &imageIndex;

    result = _brain.presentQueue.presentKHR(&presentInfo);

    _brain.device.waitIdle();

    if(result == vk::Result::eErrorOutOfDateKHR || result == vk::Result::eSuboptimalKHR || _swapChain->GetImageSize() != _application->DisplaySize())
    {
        _swapChain->RecreateSwapChain(_application->DisplaySize(), _brain.queueFamilyIndices);
    }
    else
    {
        util::VK_ASSERT(result, "Failed acquiring next image from swap chain!");
    }

    _currentFrame = (_currentFrame + 1) % MAX_FRAMES_IN_FLIGHT;

    _performanceTracker.Update();
}

Engine::~Engine()
{
    _application->ShutdownImGui();
    ImGui_ImplVulkan_Shutdown();
    ImPlot::DestroyContext();
    ImGui::DestroyContext();

    for(size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i)
    {
        _brain.device.destroy(_inFlightFences[i]);
        _brain.device.destroy(_renderFinishedSemaphores[i]);
        _brain.device.destroy(_imageAvailableSemaphores[i]);
    }

    for(auto& mesh : _model.meshes)
    {
        for(auto& primitive : mesh.primitives)
        {
            vmaDestroyBuffer(_brain.vmaAllocator, primitive.vertexBuffer, primitive.vertexBufferAllocation);
            vmaDestroyBuffer(_brain.vmaAllocator, primitive.indexBuffer, primitive.indexBufferAllocation);
        }
    }
    for(auto& texture : _model.textures)
    {
        _brain.device.destroy(texture->imageView);
        vmaDestroyImage(_brain.vmaAllocator, texture->image, texture->imageAllocation);
    }
    for(auto& material : _model.materials)
    {
        vmaDestroyBuffer(_brain.vmaAllocator, material->materialUniformBuffer, material->materialUniformAllocation);
    }

    vmaDestroyBuffer(_brain.vmaAllocator, _defaultMaterial.materialUniformBuffer, _defaultMaterial.materialUniformAllocation);
    for(auto& texture : _defaultMaterial.textures)
    {
        vmaDestroyImage(_brain.vmaAllocator, texture->image, texture->imageAllocation);
        _brain.device.destroy(texture->imageView);
    }
    _brain.device.destroy(_sampler);

    _swapChain.reset();

    _brain.device.destroy(_geometryPipeline);
    _brain.device.destroy(_geometryPipelineLayout);
    _brain.device.destroy(_lightingPipeline);
    _brain.device.destroy(_lightingPipelineLayout);

    for(size_t i = 0; i < _frameData.size(); ++i)
    {
        vmaUnmapMemory(_brain.vmaAllocator, _frameData[i].uniformBufferAllocation);
        vmaDestroyBuffer(_brain.vmaAllocator, _frameData[i].uniformBuffer, _frameData[i].uniformBufferAllocation);

        vmaDestroyImage(_brain.vmaAllocator, _frameData[i].gBuffersImageArray, _frameData[i].gBufferAllocation);
        for(size_t j = 0; j < _frameData[i].gBufferViews.size(); ++j)
        {
            _brain.device.destroy(_frameData[i].gBufferViews[j]);
        }
    }

    _brain.device.destroy(_geometryDescriptorSetLayout);
    _brain.device.destroy(_lightingDescriptorSetLayout);
    _brain.device.destroy(_materialDescriptorSetLayout);
}

void Engine::CreateGeometryPipeline()
{
    vk::PipelineLayoutCreateInfo pipelineLayoutCreateInfo{};
    std::array<vk::DescriptorSetLayout, 2> layouts = { _geometryDescriptorSetLayout, _materialDescriptorSetLayout };
    pipelineLayoutCreateInfo.setLayoutCount = layouts.size();
    pipelineLayoutCreateInfo.pSetLayouts = layouts.data();
    pipelineLayoutCreateInfo.pushConstantRangeCount = 0;
    pipelineLayoutCreateInfo.pPushConstantRanges = nullptr;

    util::VK_ASSERT(_brain.device.createPipelineLayout(&pipelineLayoutCreateInfo, nullptr, &_geometryPipelineLayout),
                    "Failed creating geometry pipeline layout!");

    auto vertByteCode = shader::ReadFile("shaders/geom-v.spv");
    auto fragByteCode = shader::ReadFile("shaders/geom-f.spv");

    vk::ShaderModule vertModule = shader::CreateShaderModule(vertByteCode, _brain.device);
    vk::ShaderModule fragModule = shader::CreateShaderModule(fragByteCode, _brain.device);

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

    std::array<vk::PipelineColorBlendAttachmentState, FrameData::DEFERRED_ATTACHMENT_COUNT> colorBlendAttachmentStates{};
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
    std::array<vk::Format, FrameData::DEFERRED_ATTACHMENT_COUNT> formats{};
    std::fill(formats.begin(), formats.end(), vk::Format::eR16G16B16A16Sfloat);
    pipelineRenderingCreateInfoKhr.colorAttachmentCount = FrameData::DEFERRED_ATTACHMENT_COUNT;
    pipelineRenderingCreateInfoKhr.pColorAttachmentFormats = formats.data();
    pipelineRenderingCreateInfoKhr.depthAttachmentFormat = _swapChain->GetDepthFormat();

    geometryPipelineCreateInfo.pNext = &pipelineRenderingCreateInfoKhr;
    geometryPipelineCreateInfo.renderPass = nullptr; // Using dynamic rendering.

    auto result = _brain.device.createGraphicsPipeline(nullptr, geometryPipelineCreateInfo, nullptr);
    util::VK_ASSERT(result.result, "Failed creating the geometry pipeline layout!");
    _geometryPipeline = result.value;

    _brain.device.destroy(vertModule);
    _brain.device.destroy(fragModule);
}

void Engine::CreateLightingPipeline()
{
    vk::PipelineLayoutCreateInfo pipelineLayoutCreateInfo{};
    pipelineLayoutCreateInfo.setLayoutCount = 1;
    pipelineLayoutCreateInfo.pSetLayouts = &_lightingDescriptorSetLayout;
    pipelineLayoutCreateInfo.pushConstantRangeCount = 0;
    pipelineLayoutCreateInfo.pPushConstantRanges = nullptr;

    util::VK_ASSERT(_brain.device.createPipelineLayout(&pipelineLayoutCreateInfo, nullptr, &_lightingPipelineLayout),
                    "Failed creating geometry pipeline layout!");

    auto vertByteCode = shader::ReadFile("shaders/lighting-v.spv");
    auto fragByteCode = shader::ReadFile("shaders/lighting-f.spv");

    vk::ShaderModule vertModule = shader::CreateShaderModule(vertByteCode, _brain.device);
    vk::ShaderModule fragModule = shader::CreateShaderModule(fragByteCode, _brain.device);

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

    auto result = _brain.device.createGraphicsPipeline(nullptr, lightingPipelineCreateInfo, nullptr);
    util::VK_ASSERT(result.result, "Failed creating the geometry pipeline layout!");
    _lightingPipeline = result.value;

    _brain.device.destroy(vertModule);
    _brain.device.destroy(fragModule);
}

void Engine::CreateCommandBuffers()
{
    vk::CommandBufferAllocateInfo commandBufferAllocateInfo{};
    commandBufferAllocateInfo.commandPool = _brain.commandPool;
    commandBufferAllocateInfo.level = vk::CommandBufferLevel::ePrimary;
    commandBufferAllocateInfo.commandBufferCount = _commandBuffers.size();

    util::VK_ASSERT(_brain.device.allocateCommandBuffers(&commandBufferAllocateInfo, _commandBuffers.data()),
                    "Failed allocating command buffer!");
}

void Engine::RecordCommandBuffer(const vk::CommandBuffer &commandBuffer, uint32_t swapChainImageIndex)
{
    vk::CommandBufferBeginInfo commandBufferBeginInfo{};
    util::VK_ASSERT(commandBuffer.begin(&commandBufferBeginInfo), "Failed to begin recording command buffer!");

    util::TransitionImageLayout(commandBuffer, _swapChain->GetImage(swapChainImageIndex), _swapChain->GetFormat(), vk::ImageLayout::eUndefined, vk::ImageLayout::eColorAttachmentOptimal);

    util::TransitionImageLayout(commandBuffer, _frameData[_currentFrame].gBuffersImageArray,
                                vk::Format::eR16G16B16A16Sfloat, vk::ImageLayout::eUndefined, vk::ImageLayout::eColorAttachmentOptimal,
                                FrameData::DEFERRED_ATTACHMENT_COUNT);

    std::array<vk::RenderingAttachmentInfoKHR, FrameData::DEFERRED_ATTACHMENT_COUNT> colorAttachmentInfos{};
    for(size_t i = 0; i < colorAttachmentInfos.size(); ++i)
    {
        vk::RenderingAttachmentInfoKHR& info{ colorAttachmentInfos[i] };
        info.imageView = _frameData[_currentFrame].gBufferViews[i];
        info.imageLayout = vk::ImageLayout::eColorAttachmentOptimal;
        info.storeOp = vk::AttachmentStoreOp::eStore;
        info.loadOp = vk::AttachmentLoadOp::eClear;
        info.clearValue.color = vk::ClearColorValue{ 0.0f, 0.0f, 0.0f, 0.0f };
    }

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

    util::BeginLabel(commandBuffer, "Geometry pass", glm::vec3{ 6.0f, 214.0f, 160.0f } / 255.0f, _brain.dldi);

    commandBuffer.beginRenderingKHR(&renderingInfo, _brain.dldi);

    commandBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, _geometryPipeline);

    commandBuffer.setViewport(0, 1, &_viewport);
    commandBuffer.setScissor(0, 1, &_scissor);

    for(auto& mesh : _model.meshes)
    {
        for(auto& primitive : mesh.primitives)
        {
            if(primitive.topology != vk::PrimitiveTopology::eTriangleList)
                throw std::runtime_error("No support for topology other than triangle list!");

            const MaterialHandle& material = primitive.material != nullptr ? *primitive.material : _defaultMaterial;

            vk::Buffer vertexBuffers[] = { primitive.vertexBuffer };
            vk::DeviceSize offsets[] = { 0 };

            commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, _geometryPipelineLayout, 0, 1, &_frameData[_currentFrame].geometryDescriptorSet, 0, nullptr);
            commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, _geometryPipelineLayout, 1, 1, &material.descriptorSet, 0, nullptr);

            commandBuffer.bindVertexBuffers(0, 1, vertexBuffers, offsets);
            commandBuffer.bindIndexBuffer(primitive.indexBuffer, 0, primitive.indexType);

            commandBuffer.drawIndexed(primitive.triangleCount, 1, 0, 0, 0);
        }
    }

    commandBuffer.endRenderingKHR(_brain.dldi);

    util::EndLabel(commandBuffer, _brain.dldi);


    util::TransitionImageLayout(commandBuffer, _frameData[_currentFrame].gBuffersImageArray,
                                vk::Format::eR16G16B16A16Sfloat, vk::ImageLayout::eColorAttachmentOptimal, vk::ImageLayout::eShaderReadOnlyOptimal,
                                FrameData::DEFERRED_ATTACHMENT_COUNT);

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

    util::BeginLabel(commandBuffer, "Lighting pass", glm::vec3{ 255.0f, 209.0f, 102.0f } / 255.0f, _brain.dldi);
    commandBuffer.beginRenderingKHR(&renderingInfo, _brain.dldi);

    commandBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, _lightingPipeline);

    commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, _lightingPipelineLayout, 0, 1, &_frameData[_currentFrame].lightingDescriptorSet, 0, nullptr);

    // Fullscreen quad.
    commandBuffer.draw(3, 1, 0, 0);

    ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), _commandBuffers[_currentFrame]);

    commandBuffer.endRenderingKHR(_brain.dldi);
    util::EndLabel(commandBuffer, _brain.dldi);

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
        util::VK_ASSERT(_brain.device.createSemaphore(&semaphoreCreateInfo, nullptr, &_imageAvailableSemaphores[i]), errorMsg);
        util::VK_ASSERT(_brain.device.createSemaphore(&semaphoreCreateInfo, nullptr, &_renderFinishedSemaphores[i]), errorMsg);
        util::VK_ASSERT(_brain.device.createFence(&fenceCreateInfo, nullptr, &_inFlightFences[i]), errorMsg);
    }
}

void Engine::CreateBuffer(vk::DeviceSize size, vk::BufferUsageFlags usage, vk::Buffer& buffer, bool mappable, VmaAllocation& allocation, std::string_view name) const
{
    vk::BufferCreateInfo bufferInfo{};
    bufferInfo.size = size;
    bufferInfo.usage = usage;
    bufferInfo.sharingMode = vk::SharingMode::eExclusive;
    bufferInfo.queueFamilyIndexCount = 1;
    bufferInfo.pQueueFamilyIndices = &_brain.queueFamilyIndices.graphicsFamily.value();

    VmaAllocationCreateInfo allocationInfo{};
    allocationInfo.usage = VMA_MEMORY_USAGE_AUTO;
    if(mappable)
        allocationInfo.flags |= VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT;

    util::VK_ASSERT(vmaCreateBuffer(_brain.vmaAllocator, reinterpret_cast<VkBufferCreateInfo*>(&bufferInfo), &allocationInfo, reinterpret_cast<VkBuffer*>(&buffer), &allocation, nullptr), "Failed creating buffer!");
    vmaSetAllocationName(_brain.vmaAllocator, allocation, name.data());
}

template <typename T>
void Engine::CreateLocalBuffer(const std::vector<T>& vec, vk::Buffer& buffer, VmaAllocation& allocation, vk::BufferUsageFlags usage, std::string_view name) const
{
    vk::DeviceSize bufferSize = vec.size() * sizeof(T);

    vk::Buffer stagingBuffer;
    VmaAllocation stagingBufferAllocation;
    CreateBuffer(bufferSize, vk::BufferUsageFlagBits::eTransferSrc, stagingBuffer, true, stagingBufferAllocation, "Staging buffer");

    vmaCopyMemoryToAllocation(_brain.vmaAllocator, vec.data(), stagingBufferAllocation, 0, bufferSize);

    CreateBuffer(bufferSize, vk::BufferUsageFlagBits::eTransferDst | usage, buffer, false, allocation, name.data());

    CopyBuffer(stagingBuffer, buffer, bufferSize);
    _brain.device.destroy(stagingBuffer, nullptr);
    vmaFreeMemory(_brain.vmaAllocator, stagingBufferAllocation);
}

void Engine::CopyBuffer(vk::Buffer srcBuffer, vk::Buffer dstBuffer, vk::DeviceSize size) const
{
    vk::CommandBuffer commandBuffer = util::BeginSingleTimeCommands(_brain.device, _brain.commandPool);

    vk::BufferCopy copyRegion{};
    copyRegion.srcOffset = 0;
    copyRegion.dstOffset = 0;
    copyRegion.size = size;
    commandBuffer.copyBuffer(srcBuffer, dstBuffer, 1, &copyRegion);

    util::EndSingleTimeCommands(_brain.device, _brain.graphicsQueue, commandBuffer, _brain.commandPool);
}

void Engine::CreateUniformBuffers()
{
    vk::DeviceSize bufferSize = sizeof(UBO);

    for(size_t i = 0; i < _frameData.size(); ++i)
    {
        CreateBuffer(bufferSize,
                     vk::BufferUsageFlagBits::eUniformBuffer,
                     _frameData[i].uniformBuffer, true, _frameData[i].uniformBufferAllocation,
                     "Uniform buffer");

        util::VK_ASSERT(vmaMapMemory(_brain.vmaAllocator, _frameData[i].uniformBufferAllocation, &_frameData[i].uniformBufferMapped), "Failed mapping memory for UBO!");
    }
}

void Engine::UpdateUniformData(uint32_t currentFrame)
{
    static auto startTime = std::chrono::high_resolution_clock::now();

    auto currentTime = std::chrono::high_resolution_clock::now();
    float time = std::chrono::duration<float, std::chrono::seconds::period>(currentTime - startTime).count();

    UBO ubo{};
    ubo.model = glm::rotate(glm::mat4{1.0f}, time * glm::radians(90.0f), glm::vec3{0.0f, 1.0f, 0.0f});
    ubo.view = glm::lookAt(glm::vec3{0.7f}, glm::vec3{0.0f}, glm::vec3{0.0f, 1.0f, 0.0f});
    ubo.proj = glm::perspective(glm::radians(45.0f), _swapChain->GetExtent().width / static_cast<float>(_swapChain->GetExtent().height), 0.01f, 100.0f);
    ubo.proj[1][1] *= -1;

    memcpy(_frameData[currentFrame].uniformBufferMapped, &ubo, sizeof(ubo));
}

void Engine::CreateDescriptorSetLayout()
{
    // Geometry
    {
        std::array<vk::DescriptorSetLayoutBinding, 1> bindings{};

        vk::DescriptorSetLayoutBinding& descriptorSetLayoutBinding{bindings[0]};
        descriptorSetLayoutBinding.binding = 0;
        descriptorSetLayoutBinding.descriptorCount = 1;
        descriptorSetLayoutBinding.descriptorType = vk::DescriptorType::eUniformBuffer;
        descriptorSetLayoutBinding.stageFlags = vk::ShaderStageFlagBits::eVertex;
        descriptorSetLayoutBinding.pImmutableSamplers = nullptr;

        vk::DescriptorSetLayoutCreateInfo createInfo{};
        createInfo.bindingCount = bindings.size();
        createInfo.pBindings = bindings.data();

        util::VK_ASSERT(_brain.device.createDescriptorSetLayout(&createInfo, nullptr, &_geometryDescriptorSetLayout),
                        "Failed creating geometry descriptor set layout!");
    }

    // Lighting
    {
        std::array<vk::DescriptorSetLayoutBinding, FrameData::DEFERRED_ATTACHMENT_COUNT + 1> bindings{};

        vk::DescriptorSetLayoutBinding& samplerLayoutBinding{bindings[0]};
        samplerLayoutBinding.binding = 0;
        samplerLayoutBinding.descriptorCount = 1;
        samplerLayoutBinding.descriptorType = vk::DescriptorType::eSampler;
        samplerLayoutBinding.stageFlags = vk::ShaderStageFlagBits::eFragment;
        samplerLayoutBinding.pImmutableSamplers = nullptr;

        for(size_t i = 1; i < bindings.size(); ++i)
        {
            vk::DescriptorSetLayoutBinding& binding{bindings[i]};
            binding.binding = i;
            binding.descriptorCount = 1;
            binding.descriptorType = vk::DescriptorType::eSampledImage;
            binding.stageFlags = vk::ShaderStageFlagBits::eFragment;
            binding.pImmutableSamplers = nullptr;
        }

        vk::DescriptorSetLayoutCreateInfo createInfo{};
        createInfo.bindingCount = bindings.size();
        createInfo.pBindings = bindings.data();

        util::VK_ASSERT(_brain.device.createDescriptorSetLayout(&createInfo, nullptr, &_lightingDescriptorSetLayout),
                        "Failed creating lighting descriptor set layout!");
    }
    // Material
    {
        auto layoutBindings = MaterialHandle::GetLayoutBindings();
        vk::DescriptorSetLayoutCreateInfo createInfo{};
        createInfo.bindingCount = layoutBindings.size();
        createInfo.pBindings = layoutBindings.data();
        util::VK_ASSERT(_brain.device.createDescriptorSetLayout(&createInfo, nullptr, &_materialDescriptorSetLayout),
                        "Failed creating material descriptor set layout!");
    }
}

void Engine::CreateDescriptorSets()
{
    {
        std::array<vk::DescriptorSetLayout, MAX_FRAMES_IN_FLIGHT> layouts{};
        std::for_each(layouts.begin(), layouts.end(), [this](auto& l)
        { l = _geometryDescriptorSetLayout; });
        vk::DescriptorSetAllocateInfo allocateInfo{};
        allocateInfo.descriptorPool = _brain.descriptorPool;
        allocateInfo.descriptorSetCount = MAX_FRAMES_IN_FLIGHT;
        allocateInfo.pSetLayouts = layouts.data();

        std::array<vk::DescriptorSet, MAX_FRAMES_IN_FLIGHT> descriptorSets;

        util::VK_ASSERT(_brain.device.allocateDescriptorSets(&allocateInfo, descriptorSets.data()),
                        "Failed allocating descriptor sets!");
        for (size_t i = 0; i < descriptorSets.size(); ++i)
            _frameData[i].geometryDescriptorSet = descriptorSets[i];
    }
    {
        std::array<vk::DescriptorSetLayout, MAX_FRAMES_IN_FLIGHT> layouts{};
        std::for_each(layouts.begin(), layouts.end(), [this](auto& l) { l = _lightingDescriptorSetLayout; });
        vk::DescriptorSetAllocateInfo allocateInfo{};
        allocateInfo.descriptorPool = _brain.descriptorPool;
        allocateInfo.descriptorSetCount = MAX_FRAMES_IN_FLIGHT;
        allocateInfo.pSetLayouts = layouts.data();

        std::array<vk::DescriptorSet, MAX_FRAMES_IN_FLIGHT> descriptorSets;

        util::VK_ASSERT(_brain.device.allocateDescriptorSets(&allocateInfo, descriptorSets.data()),
                        "Failed allocating descriptor sets!");
        for (size_t i = 0; i < descriptorSets.size(); ++i)
            _frameData[i].lightingDescriptorSet = descriptorSets[i];
    }

    for(size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i)
    {
        UpdateGeometryDescriptorSet(i);
        UpdateLightingDescriptorSet(i);
    }
}

void Engine::CreateTextureImage(const Texture& texture, TextureHandle& textureHandle, vk::Format format)
{
    vk::DeviceSize imageSize = texture.width * texture.height * texture.numChannels;

    vk::Buffer stagingBuffer;
    VmaAllocation stagingBufferAllocation;

    CreateBuffer(imageSize, vk::BufferUsageFlagBits::eTransferSrc, stagingBuffer, true, stagingBufferAllocation, "Texture staging buffer");

    vmaCopyMemoryToAllocation(_brain.vmaAllocator, texture.data.data(), stagingBufferAllocation, 0, imageSize);

    util::CreateImage(_brain.vmaAllocator, texture.width, texture.height, format,
                      vk::ImageTiling::eOptimal, vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eSampled,
                      textureHandle.image, textureHandle.imageAllocation, "Texture image");

    vk::CommandBuffer commandBuffer = util::BeginSingleTimeCommands(_brain.device, _brain.commandPool);
    util::TransitionImageLayout(commandBuffer, textureHandle.image, format, vk::ImageLayout::eUndefined, vk::ImageLayout::eTransferDstOptimal);
    util::EndSingleTimeCommands(_brain.device, _brain.graphicsQueue, commandBuffer, _brain.commandPool);

    CopyBufferToImage(stagingBuffer, textureHandle.image, texture.width, texture.height);

    commandBuffer = util::BeginSingleTimeCommands(_brain.device, _brain.commandPool);
    util::TransitionImageLayout(commandBuffer, textureHandle.image, format, vk::ImageLayout::eTransferDstOptimal, vk::ImageLayout::eShaderReadOnlyOptimal);
    util::EndSingleTimeCommands(_brain.device, _brain.graphicsQueue, commandBuffer, _brain.commandPool);

    _brain.device.destroy(stagingBuffer, nullptr);
    vmaFreeMemory(_brain.vmaAllocator, stagingBufferAllocation);

    textureHandle.imageView = util::CreateImageView(_brain.device, textureHandle.image, texture.GetFormat(), vk::ImageAspectFlagBits::eColor);
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

    vk::CommandBuffer commandBuffer = util::BeginSingleTimeCommands(_brain.device, _brain.commandPool);

    commandBuffer.copyBufferToImage(buffer, image, vk::ImageLayout::eTransferDstOptimal, 1, &region);

    util::EndSingleTimeCommands(_brain.device, _brain.graphicsQueue, commandBuffer, _brain.commandPool);
}

void Engine::CreateTextureSampler()
{
    vk::PhysicalDeviceProperties properties{};
    _brain.physicalDevice.getProperties(&properties);

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

    util::VK_ASSERT(_brain.device.createSampler(&createInfo, nullptr, &_sampler), "Failed creating sampler!");
}

void Engine::UpdateGeometryDescriptorSet(uint32_t frameIndex)
{
    vk::DescriptorBufferInfo bufferInfo{};
    bufferInfo.buffer = _frameData[frameIndex].uniformBuffer;
    bufferInfo.offset = 0;
    bufferInfo.range = sizeof(UBO);

    std::array<vk::WriteDescriptorSet, 1> descriptorWrites{};

    vk::WriteDescriptorSet& bufferWrite{ descriptorWrites[0] };
    bufferWrite.dstSet = _frameData[frameIndex].geometryDescriptorSet;
    bufferWrite.dstBinding = 0;
    bufferWrite.dstArrayElement = 0;
    bufferWrite.descriptorType = vk::DescriptorType::eUniformBuffer;
    bufferWrite.descriptorCount = 1;
    bufferWrite.pBufferInfo = &bufferInfo;

    _brain.device.updateDescriptorSets(descriptorWrites.size(), descriptorWrites.data(), 0, nullptr);
}

void Engine::UpdateLightingDescriptorSet(uint32_t frameIndex)
{
    vk::DescriptorImageInfo samplerInfo{};
    samplerInfo.sampler = _sampler;

    std::array<vk::DescriptorImageInfo, FrameData::DEFERRED_ATTACHMENT_COUNT> imageInfo{};
    for(size_t i = 0; i < imageInfo.size(); ++i)
    {
        imageInfo[i].imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal;
        imageInfo[i].imageView = _frameData[frameIndex].gBufferViews[i];
    }

    std::array<vk::WriteDescriptorSet, FrameData::DEFERRED_ATTACHMENT_COUNT + 1> descriptorWrites{};
    descriptorWrites[0].dstSet = _frameData[frameIndex].lightingDescriptorSet;
    descriptorWrites[0].dstBinding = 0;
    descriptorWrites[0].dstArrayElement = 0;
    descriptorWrites[0].descriptorType = vk::DescriptorType::eSampler;
    descriptorWrites[0].descriptorCount = 1;
    descriptorWrites[0].pImageInfo = &samplerInfo;

    for(size_t i = 1; i < descriptorWrites.size(); ++i)
    {
        descriptorWrites[i].dstSet = _frameData[frameIndex].lightingDescriptorSet;
        descriptorWrites[i].dstBinding = i;
        descriptorWrites[i].dstArrayElement = 0;
        descriptorWrites[i].descriptorType = vk::DescriptorType::eSampledImage;
        descriptorWrites[i].descriptorCount = 1;
        descriptorWrites[i].pImageInfo = &imageInfo[i - 1];
    }

    _brain.device.updateDescriptorSets(descriptorWrites.size(), descriptorWrites.data(), 0, nullptr);
}


ModelHandle Engine::LoadModel(const Model& model)
{
    ModelHandle modelHandle{};

    // Load textures
    for(const auto& texture : model.textures)
    {
        TextureHandle textureHandle{};
        textureHandle.format = texture.GetFormat();
        textureHandle.width = texture.width;
        textureHandle.height = texture.height;
        textureHandle.numChannels = texture.numChannels;

        CreateTextureImage(texture, textureHandle, textureHandle.format);

        modelHandle.textures.emplace_back(std::make_shared<TextureHandle>(textureHandle));
    }

    // Load materials
    for(const auto& material : model.materials)
    {
        std::array<std::shared_ptr<TextureHandle>, 5> textures;
        textures[0] = material.albedoIndex.has_value() ? modelHandle.textures[material.albedoIndex.value()] : nullptr;
        textures[1] = material.metallicRoughnessIndex.has_value() ? modelHandle.textures[material.metallicRoughnessIndex.value()] : nullptr;
        textures[2] = material.normalIndex.has_value() ? modelHandle.textures[material.normalIndex.value()] : nullptr;
        textures[3] = material.occlusionIndex.has_value() ? modelHandle.textures[material.occlusionIndex.value()] : nullptr;
        textures[4] = material.emissiveIndex.has_value() ? modelHandle.textures[material.emissiveIndex.value()] : nullptr;

        MaterialHandle::MaterialInfo info;
        info.useAlbedoMap = material.albedoIndex.has_value();
        info.useMRMap = material.metallicRoughnessIndex.has_value();
        info.useNormalMap = material.normalIndex.has_value();
        info.useOcclusionMap = material.occlusionIndex.has_value();
        info.useEmissiveMap = material.emissiveIndex.has_value();

        info.albedoFactor = material.albedoFactor;
        info.metallicFactor = material.metallicFactor;
        info.roughnessFactor = material.roughnessFactor;
        info.normalScale = material.normalScale;
        info.occlusionStrength = material.occlusionStrength;
        info.emissiveFactor = material.emissiveFactor;

        modelHandle.materials.emplace_back(std::make_shared<MaterialHandle>(CreateMaterial(textures, info)));
    }

    // Load meshes
    for(const auto& mesh : model.meshes)
    {
        MeshHandle meshHandle{};

        for(const auto& primitive : mesh.primitives)
        {
            MeshPrimitiveHandle primitiveHandle{};
            primitiveHandle.material = primitive.materialIndex.has_value() ? modelHandle.materials[primitive.materialIndex.value()] : nullptr;
            primitiveHandle.topology = primitive.topology;
            primitiveHandle.indexType = primitive.indexType;
            primitiveHandle.triangleCount = primitive.indices.size() / (primitiveHandle.indexType == vk::IndexType::eUint16 ? 2 : 4);

            CreateLocalBuffer(primitive.vertices, primitiveHandle.vertexBuffer, primitiveHandle.vertexBufferAllocation, vk::BufferUsageFlagBits::eVertexBuffer, "Vertex buffer");
            CreateLocalBuffer(primitive.indices, primitiveHandle.indexBuffer, primitiveHandle.indexBufferAllocation, vk::BufferUsageFlagBits::eIndexBuffer, "Index buffer");

            meshHandle.primitives.emplace_back(primitiveHandle);
        }

        modelHandle.meshes.emplace_back(meshHandle);
    }

    return modelHandle;
}

void Engine::InitializeDeferredRTs()
{
    vk::Format format = vk::Format::eR16G16B16A16Sfloat;
    for(size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i)
    {
        uint32_t width = _swapChain->GetImageSize().x;
        uint32_t height = _swapChain->GetImageSize().y;
        util::CreateImage(_brain.vmaAllocator, width, height, format,
                          vk::ImageTiling::eOptimal, vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eSampled,
                          _frameData[i].gBuffersImageArray, _frameData[i].gBufferAllocation, "GBuffer array", FrameData::DEFERRED_ATTACHMENT_COUNT);
        util::NameObject(_frameData[i].gBuffersImageArray, "[IMAGE] GBuffer Array", _brain.device, _brain.dldi);

        for(size_t j = 0; j < FrameData::DEFERRED_ATTACHMENT_COUNT; ++j)
        {
            _frameData[i].gBufferViews[j] = util::CreateImageView(_brain.device, _frameData[i].gBuffersImageArray, format, vk::ImageAspectFlagBits::eColor, j);
            std::string name = "[VIEW] GBuffer ";
            if(i == 0) name = "RGB: Albedo A: Metallic";
            else if(i == 1) name = "RGB: Normal A: Roughness";
            else if(i == 2) name = "RGB: Emissive A: AO";
            else if(i == 3) name = "RGB: Position A: Unused";

            util::NameObject(_frameData[i].gBufferViews[j], name, _brain.device, _brain.dldi);
        }

        vk::CommandBuffer cb = util::BeginSingleTimeCommands(_brain.device, _brain.commandPool);
        util::TransitionImageLayout(cb, _frameData[i].gBuffersImageArray, format, vk::ImageLayout::eUndefined, vk::ImageLayout::eColorAttachmentOptimal, FrameData::DEFERRED_ATTACHMENT_COUNT);
        util::EndSingleTimeCommands(_brain.device, _brain.graphicsQueue, cb, _brain.commandPool);
    }
}

MaterialHandle Engine::CreateMaterial(const std::array<std::shared_ptr<TextureHandle>, 5>& textures, const MaterialHandle::MaterialInfo& info)
{
    MaterialHandle materialHandle;
    materialHandle.textures = textures;

    CreateBuffer(sizeof(MaterialHandle::MaterialInfo), vk::BufferUsageFlagBits::eUniformBuffer, materialHandle.materialUniformBuffer, true, materialHandle.materialUniformAllocation, "Material uniform buffer");

    void* uniformPtr;
    util::VK_ASSERT(vmaMapMemory(_brain.vmaAllocator, materialHandle.materialUniformAllocation, &uniformPtr), "Failed mapping memory for material UBO!");
    std::memcpy(uniformPtr, &info, sizeof(info));
    vmaUnmapMemory(_brain.vmaAllocator, materialHandle.materialUniformAllocation);


    vk::DescriptorSetAllocateInfo allocateInfo{};
    allocateInfo.descriptorPool = _brain.descriptorPool;
    allocateInfo.descriptorSetCount = 1;
    allocateInfo.pSetLayouts = &_materialDescriptorSetLayout;

    util::VK_ASSERT(_brain.device.allocateDescriptorSets(&allocateInfo, &materialHandle.descriptorSet),
                    "Failed allocating material descriptor set!");

    std::array<vk::DescriptorImageInfo, 6> imageInfos;
    imageInfos[0].sampler = _sampler;
    for(size_t i = 1; i < MaterialHandle::TEXTURE_COUNT + 1; ++i)
    {
        const MaterialHandle& material = textures[i - 1] != nullptr ? materialHandle : _defaultMaterial;

        imageInfos[i].imageView = material.textures[i - 1]->imageView;
        imageInfos[i].imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal;
    }

    vk::DescriptorBufferInfo uniformInfo{};
    uniformInfo.offset = 0;
    uniformInfo.buffer = materialHandle.materialUniformBuffer;
    uniformInfo.range = sizeof(MaterialHandle::MaterialInfo);

    std::array<vk::WriteDescriptorSet, imageInfos.size() + 1> writes;
    for(size_t i = 0; i < imageInfos.size(); ++i)
    {
        writes[i].dstSet = materialHandle.descriptorSet;
        writes[i].dstBinding = i;
        writes[i].dstArrayElement = 0;
        // Hacky way of keeping this process in one loop.
        writes[i].descriptorType = i == 0 ? vk::DescriptorType::eSampler : vk::DescriptorType::eSampledImage;
        writes[i].descriptorCount = 1;
        writes[i].pImageInfo = &imageInfos[i];
    }
    writes[imageInfos.size()].dstSet = materialHandle.descriptorSet;
    writes[imageInfos.size()].dstBinding = imageInfos.size();
    writes[imageInfos.size()].dstArrayElement = 0;
    writes[imageInfos.size()].descriptorType = vk::DescriptorType::eUniformBuffer;
    writes[imageInfos.size()].descriptorCount = 1;
    writes[imageInfos.size()].pBufferInfo = &uniformInfo;

    _brain.device.updateDescriptorSets(writes.size(), writes.data(), 0, nullptr);

    return materialHandle;
}

void Engine::CreateDefaultMaterial()
{
    Texture texture;
    texture.width = 2;
    texture.height = 2;
    texture.numChannels = 4;
    texture.data = std::vector<std::byte>(texture.width * texture.height * texture.numChannels * sizeof(float));
    std::array<std::shared_ptr<TextureHandle>, 5> textures;
    std::for_each(textures.begin(), textures.end(), [&texture, this](auto& ptr){
        ptr = std::make_shared<TextureHandle>();
        CreateTextureImage(texture, *ptr, vk::Format::eR8G8B8A8Srgb);
    });

    MaterialHandle::MaterialInfo info;
    _defaultMaterial = CreateMaterial(textures, info);
}
