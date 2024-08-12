#pragma once

#include "application.hpp"
#include "swap_chain.hpp"
#include <glm/glm.hpp>
#include "engine_init_info.hpp"
#include "performance_tracker.hpp"
#include "mesh.hpp"
#include "vulkan_brain.hpp"
#include "gbuffers.hpp"
#include "include.hpp"
#include "pipelines/geometry_pipeline.hpp"

struct FrameData
{
    vk::DescriptorSet lightingDescriptorSet;
};

class Engine
{
public:

    Engine(const InitInfo& initInfo, std::shared_ptr<Application> application);
    ~Engine();
    NON_COPYABLE(Engine);
    NON_MOVABLE(Engine);

    void Run();

private:
    const VulkanBrain _brain;
    vk::DescriptorSetLayout _lightingDescriptorSetLayout;
    vk::DescriptorSetLayout _materialDescriptorSetLayout;
    std::array<vk::CommandBuffer, MAX_FRAMES_IN_FLIGHT> _commandBuffers;

    std::unique_ptr<GeometryPipeline> _geometryPipeline;

    vk::PipelineLayout _lightingPipelineLayout;
    vk::Pipeline _lightingPipeline;

    ModelHandle _model;
    MaterialHandle _defaultMaterial;

    vk::Sampler _sampler;

    std::unique_ptr<SwapChain> _swapChain;
    std::unique_ptr<GBuffers> _gBuffers;


    std::array<vk::Semaphore, MAX_FRAMES_IN_FLIGHT> _imageAvailableSemaphores;
    std::array<vk::Semaphore, MAX_FRAMES_IN_FLIGHT> _renderFinishedSemaphores;
    std::array<vk::Fence, MAX_FRAMES_IN_FLIGHT> _inFlightFences;

    std::shared_ptr<Application> _application;

    uint32_t _currentFrame{ 0 };

    PerformanceTracker _performanceTracker;

    std::array<FrameData, MAX_FRAMES_IN_FLIGHT> _frameData;

    void CreateDescriptorSetLayout();
    void CreateGeometryPipeline();
    void CreateLightingPipeline();
    void CreateCommandBuffers();
    void RecordCommandBuffer(const vk::CommandBuffer& commandBuffer, uint32_t swapChainImageIndex);
    void CreateSyncObjects();
    void CreateTextureImage(const Texture& texture, TextureHandle& textureHandle, vk::Format format);
    void CopyBufferToImage(vk::Buffer buffer, vk::Image image, uint32_t width, uint32_t height);
    void CreateTextureSampler();
    void CreateDescriptorSets();
    void UpdateLightingDescriptorSet(uint32_t frameIndex);
    ModelHandle LoadModel(const Model& model);

    MaterialHandle CreateMaterial(const std::array<std::shared_ptr<TextureHandle>, 5>& textures, const MaterialHandle::MaterialInfo& info);
    void CreateDefaultMaterial();
};
