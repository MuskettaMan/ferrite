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
#include "pipelines/lighting_pipeline.hpp"
#include "model_loader.hpp"

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
    vk::DescriptorSetLayout _materialDescriptorSetLayout;
    std::array<vk::CommandBuffer, MAX_FRAMES_IN_FLIGHT> _commandBuffers;

    std::unique_ptr<GeometryPipeline> _geometryPipeline;
    std::unique_ptr<LightingPipeline> _lightingPipeline;
    std::unique_ptr<ModelLoader> _modelLoader;

    SceneDescription _scene;

    std::unique_ptr<SwapChain> _swapChain;
    std::unique_ptr<GBuffers> _gBuffers;

    std::array<vk::Semaphore, MAX_FRAMES_IN_FLIGHT> _imageAvailableSemaphores;
    std::array<vk::Semaphore, MAX_FRAMES_IN_FLIGHT> _renderFinishedSemaphores;
    std::array<vk::Fence, MAX_FRAMES_IN_FLIGHT> _inFlightFences;

    std::shared_ptr<Application> _application;

    uint32_t _currentFrame{ 0 };
    std::chrono::time_point<std::chrono::high_resolution_clock> _lastFrameTime;

    PerformanceTracker _performanceTracker;

    glm::vec2 _previousMousePos;

    void CreateDescriptorSetLayout();
    void CreateCommandBuffers();
    void RecordCommandBuffer(const vk::CommandBuffer& commandBuffer, uint32_t swapChainImageIndex);
    void CreateSyncObjects();
};
