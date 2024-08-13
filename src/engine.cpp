#include "engine.hpp"


#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

#define VMA_IMPLEMENTATION
#define VMA_LEAK_LOG_FORMAT(format, ...) do { \
        spdlog::info(fmt::sprintf(format, __VA_ARGS__)); \
    } while(false)
#include "vk_mem_alloc.h"


#include "include.hpp"
#include "imgui.h"
#include "vulkan_validation.hpp"
#include "vulkan_helper.hpp"
#include "shaders/shader_loader.hpp"
#include "imgui_impl_vulkan.h"
#include "stopwatch.hpp"
#include "model_loader.hpp"
#include "util.hpp"
#include "vulkan_brain.hpp"
#include <implot.h>

Engine::Engine(const InitInfo& initInfo, std::shared_ptr<Application> application) :
    _brain(initInfo)
{
    ImGui::CreateContext();
    ImPlot::CreateContext();
    spdlog::info("Starting engine...");

    _application = std::move(application);

    CreateDescriptorSetLayout();

    _swapChain = std::make_unique<SwapChain>(_brain, glm::uvec2{ initInfo.width, initInfo.height });
    _gBuffers = std::make_unique<GBuffers>(_brain, _swapChain->GetImageSize());
    _geometryPipeline = std::make_unique<GeometryPipeline>(_brain, *_gBuffers, _materialDescriptorSetLayout);
    _lightingPipeline = std::make_unique<LightingPipeline>(_brain, *_gBuffers, *_swapChain);

    CreateTextureSampler();
    CreateCommandBuffers();
    CreateSyncObjects();
    CreateDefaultMaterial();

    ModelLoader modelLoader;
    Model model = modelLoader.Load("assets/models/ABeautifulGame/ABeautifulGame.gltf");
    //Model model = modelLoader.Load("assets/models/DamagedHelmet.glb");
    _model = LoadModel(model);

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

    util::TransitionImageLayout(commandBuffer, _gBuffers->GBuffersImageArray(_currentFrame),
                                _gBuffers->GBufferFormat(), vk::ImageLayout::eUndefined, vk::ImageLayout::eColorAttachmentOptimal,
                                DEFERRED_ATTACHMENT_COUNT);

    _geometryPipeline->RecordCommands(commandBuffer, _currentFrame, _model);


    util::TransitionImageLayout(commandBuffer, _gBuffers->GBuffersImageArray(_currentFrame),
                                _gBuffers->GBufferFormat(), vk::ImageLayout::eColorAttachmentOptimal, vk::ImageLayout::eShaderReadOnlyOptimal,
                                DEFERRED_ATTACHMENT_COUNT);

    _lightingPipeline->RecordCommands(commandBuffer, _currentFrame, swapChainImageIndex);

    // TODO: Figure this out.
    //ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), _commandBuffers[_currentFrame]);

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

void Engine::CreateTextureImage(const Texture& texture, TextureHandle& textureHandle, vk::Format format)
{
    vk::DeviceSize imageSize = texture.width * texture.height * texture.numChannels;

    vk::Buffer stagingBuffer;
    VmaAllocation stagingBufferAllocation;

    util::CreateBuffer(_brain, imageSize, vk::BufferUsageFlagBits::eTransferSrc, stagingBuffer, true, stagingBufferAllocation, "Texture staging buffer");

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

            util::CreateLocalBuffer(_brain, primitive.vertices, primitiveHandle.vertexBuffer, primitiveHandle.vertexBufferAllocation, vk::BufferUsageFlagBits::eVertexBuffer, "Vertex buffer");
            util::CreateLocalBuffer(_brain, primitive.indices, primitiveHandle.indexBuffer, primitiveHandle.indexBufferAllocation, vk::BufferUsageFlagBits::eIndexBuffer, "Index buffer");

            meshHandle.primitives.emplace_back(primitiveHandle);
        }

        modelHandle.meshes.emplace_back(meshHandle);
    }

    return modelHandle;
}

MaterialHandle Engine::CreateMaterial(const std::array<std::shared_ptr<TextureHandle>, 5>& textures, const MaterialHandle::MaterialInfo& info)
{
    MaterialHandle materialHandle;
    materialHandle.textures = textures;

    util::CreateBuffer(_brain, sizeof(MaterialHandle::MaterialInfo), vk::BufferUsageFlagBits::eUniformBuffer, materialHandle.materialUniformBuffer, true, materialHandle.materialUniformAllocation, "Material uniform buffer");

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
