#include "engine.hpp"


#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

#define VMA_IMPLEMENTATION
#define VMA_LEAK_LOG_FORMAT(format, ...) do { \
        spdlog::info(fmt::sprintf(format, __VA_ARGS__)); \
    } while(false)
#include "vk_mem_alloc.h"


#include "include.hpp"
#include "vulkan_validation.hpp"
#include "vulkan_helper.hpp"
#include "imgui_impl_vulkan.h"
#include "stopwatch.hpp"
#include "model_loader.hpp"
#include "util.hpp"
#include "mesh_primitives.hpp"
#include "pipelines/geometry_pipeline.hpp"
#include "pipelines/lighting_pipeline.hpp"
#include "pipelines/skydome_pipeline.hpp"
#include "pipelines/tonemapping_pipeline.hpp"
#include "pipelines/ibl_pipeline.hpp"
#include "gbuffers.hpp"
#include "application.hpp"
#include "single_time_commands.hpp"

Engine::Engine(const InitInfo& initInfo, std::shared_ptr<Application> application) :
    _brain(initInfo)
{
    ImGui::CreateContext();
    ImPlot::CreateContext();
    spdlog::info("Starting engine...");

    _application = std::move(application);

    _swapChain = std::make_unique<SwapChain>(_brain, glm::uvec2{ initInfo.width, initInfo.height });

    CreateDescriptorSetLayout();
    InitializeCameraUBODescriptors();
    InitializeHDRTarget();
    LoadEnvironmentMap();

    _modelLoader = std::make_unique<ModelLoader>(_brain, _materialDescriptorSetLayout);

    SingleTimeCommands commandBufferPrimitive{ _brain };
    MeshPrimitiveHandle uvSphere = _modelLoader->LoadPrimitive(GenerateUVSphere(32, 32), commandBufferPrimitive);
    commandBufferPrimitive.Submit();

    _gBuffers = std::make_unique<GBuffers>(_brain, _swapChain->GetImageSize());
    _geometryPipeline = std::make_unique<GeometryPipeline>(_brain, *_gBuffers, _materialDescriptorSetLayout, _cameraStructure);
    _skydomePipeline = std::make_unique<SkydomePipeline>(_brain, std::move(uvSphere), _cameraStructure, _hdrTarget, _environmentMap);
    _tonemappingPipeline = std::make_unique<TonemappingPipeline>(_brain, _hdrTarget, *_swapChain);
    _iblPipeline = std::make_unique<IBLPipeline>(_brain, _environmentMap);
    _lightingPipeline = std::make_unique<LightingPipeline>(_brain, *_gBuffers, _hdrTarget, _cameraStructure, _iblPipeline->IrradianceMap());

    SingleTimeCommands commandBufferIBL{ _brain };
    _iblPipeline->RecordCommands(commandBufferIBL.CommandBuffer());
    commandBufferIBL.Submit();

    CreateCommandBuffers();
    CreateSyncObjects();

    //_scene.model = _modelLoader->Load("assets/models/EnvironmentTest/EnvironmentTest.gltf");
    _scene.model = _modelLoader->Load("assets/models/ABeautifulGame/ABeautifulGame.gltf");
    //_scene.model = _modelLoader->Load("assets/models/DamagedHelmet.glb");

    vk::Format format = _swapChain->GetFormat();
    vk::PipelineRenderingCreateInfoKHR pipelineRenderingCreateInfoKhr{};
    pipelineRenderingCreateInfoKhr.colorAttachmentCount = 1;
    pipelineRenderingCreateInfoKhr.pColorAttachmentFormats = &format;
    pipelineRenderingCreateInfoKhr.depthAttachmentFormat = _gBuffers->DepthFormat();

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

    _scene.camera.position = glm::vec3{ 0.0f, 0.2f, 0.0f };
    _scene.camera.fov = glm::radians(45.0f);
    _scene.camera.nearPlane = 0.01f;
    _scene.camera.farPlane = 100.0f;

    _lastFrameTime = std::chrono::high_resolution_clock::now();

    spdlog::info("Successfully initialized engine!");
}

void Engine::Run()
{
    auto currentFrameTime = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float, std::milli> deltaTime = currentFrameTime - _lastFrameTime;
    _lastFrameTime = currentFrameTime;
    float deltaTimeMS = deltaTime.count();

    // Slow down application when minimized.
    if(_application->IsMinimized())
    {
        using namespace std::chrono_literals;
        std::this_thread::sleep_for(16ms);
        return;
    }

    glm::vec2 mousePos = _application->GetMousePosition();
    glm::vec2 lastMousePos = _application->GetLastMousePosition();
    float yaw = (mousePos.x - lastMousePos.x) * -0.1f;
    float pitch = (lastMousePos.y - mousePos.y) * 0.1f;
    pitch = std::clamp(pitch, -89.0f, 89.0f);

    glm::quat yawQuat = glm::angleAxis(glm::radians(yaw), glm::vec3(0.0f, 1.0f, 0.0f));
    glm::quat pitchQuat = glm::angleAxis(glm::radians(pitch), glm::vec3(1.0f, 0.0f, 0.0f));
    glm::quat rollQuat = glm::angleAxis(glm::radians(0.0f), glm::vec3(0.0f, 0.0f, 1.0f));

    _scene.camera.rotation = yawQuat * _scene.camera.rotation;
    _scene.camera.rotation = _scene.camera.rotation * pitchQuat;
    _scene.camera.rotation = rollQuat * _scene.camera.rotation;

    _scene.camera.rotation = glm::normalize(_scene.camera.rotation);

    const float speed = 0.001f * deltaTimeMS;
    glm::vec3 movement{ 0.0f };
    if(_application->KeyPressed('W'))
        movement += glm::vec3{0.0f, 0.0f, -1.0f};
    else if(_application->KeyPressed('S'))
        movement += glm::vec3{0.0f, 0.0f, 1.0f};
    else if(_application->KeyPressed('A'))
        movement += glm::vec3{-1.0f, 0.0f, 0.0f};
    else if(_application->KeyPressed('D'))
        movement += glm::vec3{1.0f, 0.0f, 0.0f};

    _scene.camera.position += _scene.camera.rotation * movement * deltaTimeMS * speed;

    CameraUBO cameraUBO = CalculateCamera(_scene.camera);
    std::memcpy(_cameraStructure.mappedPtrs[_currentFrame], &cameraUBO, sizeof(CameraUBO));

    util::VK_ASSERT(_brain.device.waitForFences(1, &_inFlightFences[_currentFrame], vk::True, std::numeric_limits<uint64_t>::max()),
                    "Failed waiting on in flight fence!");

    uint32_t imageIndex;
    vk::Result result = _brain.device.acquireNextImageKHR(_swapChain->GetSwapChain(), std::numeric_limits<uint64_t>::max(),
                                                    _imageAvailableSemaphores[_currentFrame], nullptr, &imageIndex);

    if(result == vk::Result::eErrorOutOfDateKHR || result == vk::Result::eSuboptimalKHR)
    {
        _swapChain->Resize(_application->DisplaySize());
        _gBuffers->Resize(_application->DisplaySize());
        _lightingPipeline->UpdateGBufferViews();

        return;
    } else
        util::VK_ASSERT(result, "Failed acquiring next image from swap chain!");

    util::VK_ASSERT(_brain.device.resetFences(1, &_inFlightFences[_currentFrame]), "Failed resetting fences!");

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
        _swapChain->Resize(_application->DisplaySize());
        _gBuffers->Resize(_application->DisplaySize());
        _lightingPipeline->UpdateGBufferViews();
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

    _brain.device.destroy(_environmentMap.imageView);
    vmaDestroyImage(_brain.vmaAllocator, _environmentMap.image, _environmentMap.imageAllocation);

    for(size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i)
    {
        _brain.device.destroy(_inFlightFences[i]);
        _brain.device.destroy(_renderFinishedSemaphores[i]);
        _brain.device.destroy(_imageAvailableSemaphores[i]);
    }

    for(auto& mesh : _scene.model.meshes)
    {
        for(auto& primitive : mesh->primitives)
        {
            vmaDestroyBuffer(_brain.vmaAllocator, primitive.vertexBuffer, primitive.vertexBufferAllocation);
            vmaDestroyBuffer(_brain.vmaAllocator, primitive.indexBuffer, primitive.indexBufferAllocation);
        }
    }
    for(auto& texture : _scene.model.textures)
    {
        _brain.device.destroy(texture->imageView);
        vmaDestroyImage(_brain.vmaAllocator, texture->image, texture->imageAllocation);
    }
    for(auto& material : _scene.model.materials)
    {
        vmaDestroyBuffer(_brain.vmaAllocator, material->materialUniformBuffer, material->materialUniformAllocation);
    }

    _brain.device.destroy(_hdrTarget.imageViews);
    vmaDestroyImage(_brain.vmaAllocator, _hdrTarget.images, _hdrTarget.allocations);

    _brain.device.destroy(_cameraStructure.descriptorSetLayout);
    for(size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i)
    {
        vmaUnmapMemory(_brain.vmaAllocator, _cameraStructure.allocations[i]);
        vmaDestroyBuffer(_brain.vmaAllocator, _cameraStructure.buffers[i], _cameraStructure.allocations[i]);
    }

    _swapChain.reset();

    _brain.device.destroy(_materialDescriptorSetLayout);
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
    util::TransitionImageLayout(commandBuffer, _hdrTarget.images, _hdrTarget.format, vk::ImageLayout::eUndefined, vk::ImageLayout::eColorAttachmentOptimal);
    util::TransitionImageLayout(commandBuffer, _gBuffers->GBuffersImageArray(),
                                _gBuffers->GBufferFormat(), vk::ImageLayout::eUndefined, vk::ImageLayout::eColorAttachmentOptimal,
                                DEFERRED_ATTACHMENT_COUNT);

    _geometryPipeline->RecordCommands(commandBuffer, _currentFrame, _scene);


    util::TransitionImageLayout(commandBuffer, _gBuffers->GBuffersImageArray(),
                                _gBuffers->GBufferFormat(), vk::ImageLayout::eColorAttachmentOptimal, vk::ImageLayout::eShaderReadOnlyOptimal,
                                DEFERRED_ATTACHMENT_COUNT);

    _skydomePipeline->RecordCommands(commandBuffer, _currentFrame);
    _lightingPipeline->RecordCommands(commandBuffer, _currentFrame);

    util::TransitionImageLayout(commandBuffer, _hdrTarget.images, _hdrTarget.format, vk::ImageLayout::eColorAttachmentOptimal, vk::ImageLayout::eShaderReadOnlyOptimal);

    _tonemappingPipeline->RecordCommands(commandBuffer, _currentFrame, swapChainImageIndex);

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

void Engine::CreateDescriptorSetLayout()
{
    auto materialLayoutBindings = MaterialHandle::GetLayoutBindings();
    vk::DescriptorSetLayoutCreateInfo materialCreateInfo{};
    materialCreateInfo.bindingCount = materialLayoutBindings.size();
    materialCreateInfo.pBindings = materialLayoutBindings.data();
    util::VK_ASSERT(_brain.device.createDescriptorSetLayout(&materialCreateInfo, nullptr, &_materialDescriptorSetLayout),
                    "Failed creating material descriptor set layout!");


    vk::DescriptorSetLayoutBinding cameraUBODescriptorSetBinding{};
    cameraUBODescriptorSetBinding.binding = 0;
    cameraUBODescriptorSetBinding.descriptorType = vk::DescriptorType::eUniformBuffer;
    cameraUBODescriptorSetBinding.descriptorCount = 1;
    cameraUBODescriptorSetBinding.stageFlags = vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment;

    vk::DescriptorSetLayoutCreateInfo cameraUBOCreateInfo{};
    cameraUBOCreateInfo.bindingCount = 1;
    cameraUBOCreateInfo.pBindings = &cameraUBODescriptorSetBinding;
    util::VK_ASSERT(_brain.device.createDescriptorSetLayout(&cameraUBOCreateInfo, nullptr, &_cameraStructure.descriptorSetLayout),
                    "Failed creating camera UBO descriptor set layout!");
}

void Engine::InitializeCameraUBODescriptors()
{
    vk::DeviceSize bufferSize = sizeof(CameraUBO);

    // Create buffers.
    for(size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i)
    {
        util::CreateBuffer(_brain, bufferSize,
                           vk::BufferUsageFlagBits::eUniformBuffer,
                           _cameraStructure.buffers[i], true, _cameraStructure.allocations[i],
                           VMA_MEMORY_USAGE_CPU_ONLY,
                           "Uniform buffer");

        util::VK_ASSERT(vmaMapMemory(_brain.vmaAllocator, _cameraStructure.allocations[i], &_cameraStructure.mappedPtrs[i]), "Failed mapping memory for UBO!");
    }

    std::array<vk::DescriptorSetLayout, MAX_FRAMES_IN_FLIGHT> layouts{};
    std::for_each(layouts.begin(), layouts.end(), [this](auto& l)
    { l = _cameraStructure.descriptorSetLayout; });
    vk::DescriptorSetAllocateInfo allocateInfo{};
    allocateInfo.descriptorPool = _brain.descriptorPool;
    allocateInfo.descriptorSetCount = MAX_FRAMES_IN_FLIGHT;
    allocateInfo.pSetLayouts = layouts.data();

    util::VK_ASSERT(_brain.device.allocateDescriptorSets(&allocateInfo, _cameraStructure.descriptorSets.data()),
                    "Failed allocating descriptor sets!");

    for (size_t i = 0; i < _cameraStructure.descriptorSets.size(); ++i)
    {
        UpdateCameraDescriptorSet(i);
    }
}

void Engine::UpdateCameraDescriptorSet(uint32_t currentFrame)
{
    vk::DescriptorBufferInfo bufferInfo{};
    bufferInfo.buffer = _cameraStructure.buffers[currentFrame];
    bufferInfo.offset = 0;
    bufferInfo.range = sizeof(CameraUBO);

    std::array<vk::WriteDescriptorSet, 1> descriptorWrites{};

    vk::WriteDescriptorSet& bufferWrite{ descriptorWrites[0] };
    bufferWrite.dstSet = _cameraStructure.descriptorSets[currentFrame];
    bufferWrite.dstBinding = 0;
    bufferWrite.dstArrayElement = 0;
    bufferWrite.descriptorType = vk::DescriptorType::eUniformBuffer;
    bufferWrite.descriptorCount = 1;
    bufferWrite.pBufferInfo = &bufferInfo;

    _brain.device.updateDescriptorSets(descriptorWrites.size(), descriptorWrites.data(), 0, nullptr);
}

CameraUBO Engine::CalculateCamera(const Camera& camera)
{
    CameraUBO ubo{};

    glm::mat4 cameraRotation = glm::toMat4(camera.rotation);
    glm::mat4 cameraTranslation = glm::translate(glm::mat4{1.0f}, camera.position);

    ubo.view = glm::inverse(cameraTranslation * cameraRotation);
    ubo.proj = glm::perspective(camera.fov, _gBuffers->Size().x / static_cast<float>(_gBuffers->Size().y), camera.nearPlane, camera.farPlane);
    ubo.proj[1][1] *= -1;

    ubo.VP = ubo.proj * ubo.view;
    ubo.cameraPosition = camera.position;

    return ubo;
}

void Engine::InitializeHDRTarget()
{
    _hdrTarget.format = vk::Format::eR32G32B32A32Sfloat;
    _hdrTarget.size = _swapChain->GetImageSize();

    util::CreateImage(_brain.vmaAllocator, _hdrTarget.size.x, _hdrTarget.size.y, _hdrTarget.format,
                      vk::ImageTiling::eOptimal, vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eSampled,
                      _hdrTarget.images, _hdrTarget.allocations, "HDR Target", false, VMA_MEMORY_USAGE_GPU_ONLY);
    util::NameObject(_hdrTarget.images, "[IMAGE] HDR Target", _brain.device, _brain.dldi);

    _hdrTarget.imageViews = util::CreateImageView(_brain.device, _hdrTarget.images, _hdrTarget.format, vk::ImageAspectFlagBits::eColor);
    util::NameObject(_hdrTarget.imageViews, "HDR Target View", _brain.device, _brain.dldi);

    vk::CommandBuffer cb = util::BeginSingleTimeCommands(_brain);
    util::TransitionImageLayout(cb, _hdrTarget.images, _hdrTarget.format, vk::ImageLayout::eUndefined, vk::ImageLayout::eColorAttachmentOptimal);
    util::EndSingleTimeCommands(_brain,cb);
}

void Engine::LoadEnvironmentMap()
{
    int32_t width, height, numChannels;
    float* data = stbi_loadf("assets/hdri/industrial_sunset_02_puresky_4k.hdr", &width, &height, &numChannels, 4);

    Texture texture{};
    texture.width = width;
    texture.height = height;
    texture.numChannels = 4;
    texture.isHDR = true;
    texture.data.resize(width * height * texture.numChannels * sizeof(float));
    std::memcpy(texture.data.data(), data, texture.data.size());

    stbi_image_free(data);

    SingleTimeCommands commandBuffer{ _brain };
    commandBuffer.CreateTextureImage(texture, _environmentMap, false);
    commandBuffer.Submit();

    util::NameObject(_environmentMap.image, "Environment HDRI", _brain.device, _brain.dldi);
}
