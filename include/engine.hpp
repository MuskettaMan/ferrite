#pragma once

#include <memory>
#include <optional>
#include <functional>
#include <chrono>
#include <vulkan/vulkan.hpp>
#include "class_decorations.hpp"
#include "application.hpp"
#include "swap_chain.hpp"
#include <glm/glm.hpp>
#include "engine_init_info.hpp"
#include "performance_tracker.hpp"
#include "mesh.hpp"
#include "vulkan_brain.hpp"

struct UBO
{
    alignas(16) glm::mat4 model;
    alignas(16) glm::mat4 view;
    alignas(16) glm::mat4 proj;
};

struct FrameData
{
    const static uint32_t DEFERRED_ATTACHMENT_COUNT = 4;

    vk::Buffer uniformBuffer;
    VmaAllocation uniformBufferAllocation;
    void* uniformBufferMapped;
    vk::DescriptorSet geometryDescriptorSet;
    vk::DescriptorSet lightingDescriptorSet;

    vk::Image gBuffersImageArray;
    VmaAllocation gBufferAllocation;
    std::array<vk::ImageView, DEFERRED_ATTACHMENT_COUNT> gBufferViews;
};

class Engine
{
public:
    const static uint32_t MAX_FRAMES_IN_FLIGHT{ 3 };

    Engine(const InitInfo& initInfo, std::shared_ptr<Application> application);
    ~Engine();
    NON_COPYABLE(Engine);
    NON_MOVABLE(Engine);

    void Run();

private:
    const VulkanBrain _brain;
    vk::DescriptorSetLayout _geometryDescriptorSetLayout;
    vk::DescriptorSetLayout _lightingDescriptorSetLayout;
    vk::DescriptorSetLayout _materialDescriptorSetLayout;
    std::array<vk::CommandBuffer, MAX_FRAMES_IN_FLIGHT> _commandBuffers;
    vk::Viewport _viewport;
    vk::Rect2D _scissor;

    vk::PipelineLayout _geometryPipelineLayout;
    vk::Pipeline _geometryPipeline;

    vk::PipelineLayout _lightingPipelineLayout;
    vk::Pipeline _lightingPipeline;

    ModelHandle _model;
    MaterialHandle _defaultMaterial;

    vk::Sampler _sampler;

    std::unique_ptr<SwapChain> _swapChain;


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
    void CreateBuffer(vk::DeviceSize size, vk::BufferUsageFlags usage, vk::Buffer& buffer, bool mappable, VmaAllocation& allocation, std::string_view name) const;
    template <typename T>
    void CreateLocalBuffer(const std::vector<T>& vec, vk::Buffer& buffer, VmaAllocation& allocation, vk::BufferUsageFlags usage, std::string_view name) const;
    void CopyBuffer(vk::Buffer srcBuffer, vk::Buffer dstBuffer, vk::DeviceSize size) const;
    void CreateUniformBuffers();
    void UpdateUniformData(uint32_t currentFrame);
    void CreateTextureImage(const Texture& texture, TextureHandle& textureHandle, vk::Format format);
    void CopyBufferToImage(vk::Buffer buffer, vk::Image image, uint32_t width, uint32_t height);
    void CreateTextureSampler();
    void CreateDescriptorSets();
    void UpdateGeometryDescriptorSet(uint32_t frameIndex);
    void UpdateLightingDescriptorSet(uint32_t frameIndex);
    ModelHandle LoadModel(const Model& model);

    void InitializeDeferredRTs();
    MaterialHandle CreateMaterial(const std::array<std::shared_ptr<TextureHandle>, 5>& textures, const MaterialHandle::MaterialInfo& info);
    void CreateDefaultMaterial();
};
