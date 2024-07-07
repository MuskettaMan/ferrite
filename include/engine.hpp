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

struct QueueFamilyIndices
{
    std::optional<uint32_t> graphicsFamily;
    std::optional<uint32_t> presentFamily;

    bool IsComplete()
    {
        return graphicsFamily.has_value() && presentFamily.has_value();
    }
};

class Engine
{
public:
    const static uint32_t MAX_FRAMES_IN_FLIGHT{ 2 };

    Engine();
    ~Engine() = default;
    NON_COPYABLE(Engine);
    NON_MOVABLE(Engine);

    void Init(const InitInfo& initInfo, std::shared_ptr<Application> application);
    void Run();
    void Shutdown();

private:
    vk::Instance _instance;
    vk::PhysicalDevice _physicalDevice;
    vk::Device _device;
    vk::Queue _graphicsQueue;
    vk::Queue _presentQueue;
    vk::SurfaceKHR _surface;
    vk::PipelineLayout _pipelineLayout;
    vk::RenderPass _renderPass;
    vk::Pipeline _pipeline;
    vk::DescriptorPool _descriptorPool;

    std::unique_ptr<SwapChain> _swapChain;

    vk::CommandPool _commandPool;
    std::array<vk::CommandBuffer, MAX_FRAMES_IN_FLIGHT> _commandBuffers;

    vk::Viewport _viewport;
    vk::Rect2D _scissor;

    std::array<vk::Semaphore, MAX_FRAMES_IN_FLIGHT> _imageAvailableSemaphores;
    std::array<vk::Semaphore, MAX_FRAMES_IN_FLIGHT> _renderFinishedSemaphores;
    std::array<vk::Fence, MAX_FRAMES_IN_FLIGHT> _inFlightFences;

    vk::DebugUtilsMessengerEXT _debugMessenger;

    std::function<void()> _newImGuiFrame;

    std::shared_ptr<Application> _application;

    uint32_t _currentFrame{ 0 };

    PerformanceTracker _performanceTracker;

    const std::vector<const char*> _validationLayers =
    {
       "VK_LAYER_KHRONOS_validation"
    };
    const bool _enableValidationLayers =
#if defined(NDEBUG)
        false;
#else
        true;
#endif
    const std::vector<const char*> _deviceExtensions =
    {
        VK_KHR_SWAPCHAIN_EXTENSION_NAME,
    };


    void CreateInstance(const InitInfo& initInfo);
    bool CheckValidationLayerSupport();
    std::vector<const char*> GetRequiredExtensions(const InitInfo& initInfo);
    void SetupDebugMessenger();
    void PickPhysicalDevice();
    uint32_t RateDeviceSuitability(const vk::PhysicalDevice& device);
    bool ExtensionsSupported(const vk::PhysicalDevice& device);
    QueueFamilyIndices FindQueueFamilies(const vk::PhysicalDevice& device);
    void CreateDevice();
    void CreateGraphicsPipeline();
    void CreateRenderPass();
    void CreateCommandPool();
    void CreateCommandBuffers();
    void RecordCommandBuffer(const vk::CommandBuffer& commandBuffer, uint32_t swapChainImageIndex);
    void CreateSyncObjects();
    void CreateDescriptorPool();

    void LogInstanceExtensions();

};
