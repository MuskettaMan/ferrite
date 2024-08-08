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
    std::optional<uint32_t> tranferFamily;

    bool IsComplete()
    {
        return graphicsFamily.has_value() && presentFamily.has_value() && tranferFamily.has_value();
    }
};

struct Vertex
{
    enum Enumeration {
        ePOSITION,
        eCOLOR,
        eTEX_COORD
    };

    glm::vec2 position;
    glm::vec3 color;
    glm::vec2 texCoord;

    static vk::VertexInputBindingDescription GetBindingDescription()
    {
        vk::VertexInputBindingDescription bindingDesc;
        bindingDesc.binding = 0;
        bindingDesc.stride = sizeof(Vertex);
        bindingDesc.inputRate = vk::VertexInputRate::eVertex;

        return bindingDesc;
    }

    static std::array<vk::VertexInputAttributeDescription, 3> GetAttributeDescriptions()
    {
        std::array<vk::VertexInputAttributeDescription, 3> attributeDescriptions{};
        attributeDescriptions[ePOSITION].binding = 0;
        attributeDescriptions[ePOSITION].location = 0;
        attributeDescriptions[ePOSITION].format = vk::Format::eR32G32Sfloat;
        attributeDescriptions[ePOSITION].offset = offsetof(Vertex, position);

        attributeDescriptions[eCOLOR].binding = 0;
        attributeDescriptions[eCOLOR].location = 1;
        attributeDescriptions[eCOLOR].format = vk::Format::eR32G32B32Sfloat;
        attributeDescriptions[eCOLOR].offset = offsetof(Vertex, color);

        attributeDescriptions[eTEX_COORD].binding = 0;
        attributeDescriptions[eTEX_COORD].location = 2;
        attributeDescriptions[eTEX_COORD].format = vk::Format::eR32G32Sfloat;
        attributeDescriptions[eTEX_COORD].offset = offsetof(Vertex, texCoord);

        return attributeDescriptions;
    }
};

struct UBO
{
    alignas(16) glm::mat4 model;
    alignas(16) glm::mat4 view;
    alignas(16) glm::mat4 proj;
};

struct FrameData
{
    vk::Buffer uniformBuffer;
    vk::DeviceMemory uniformBufferMemory;
    void* uniformBufferMapped;
    vk::DescriptorSet descriptorSet;
};

class Engine
{
public:
    const static uint32_t MAX_FRAMES_IN_FLIGHT{ 3 };

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
    vk::Queue _transferQueue;
    vk::SurfaceKHR _surface;
    vk::PipelineLayout _pipelineLayout;
    vk::Pipeline _pipeline;
    vk::DescriptorPool _descriptorPool;
    vk::DescriptorSetLayout _descriptorSetLayout;
    vk::CommandPool _commandPool;
    vk::CommandPool _transferCommandPool;
    std::array<vk::CommandBuffer, MAX_FRAMES_IN_FLIGHT> _commandBuffers;
    vk::Viewport _viewport;
    vk::Rect2D _scissor;
    vk::DispatchLoaderDynamic _dldi;

    vk::Buffer _vertexBuffer;
    vk::DeviceMemory _vertexBufferMemory;
    vk::Buffer _indexBuffer;
    vk::DeviceMemory _indexBufferMemory;
    vk::Image _image;
    vk::DeviceMemory _imageMemory;
    vk::ImageView _imageView;
    vk::Sampler _sampler;

    std::unique_ptr<SwapChain> _swapChain;

    QueueFamilyIndices _queueFamilyIndices;

    std::array<vk::Semaphore, MAX_FRAMES_IN_FLIGHT> _imageAvailableSemaphores;
    std::array<vk::Semaphore, MAX_FRAMES_IN_FLIGHT> _renderFinishedSemaphores;
    std::array<vk::Fence, MAX_FRAMES_IN_FLIGHT> _inFlightFences;

    vk::DebugUtilsMessengerEXT _debugMessenger;

    std::function<void()> _newImGuiFrame;
    std::function<void()> _shutdownImGui;

    std::shared_ptr<Application> _application;

    uint32_t _currentFrame{ 0 };

    PerformanceTracker _performanceTracker;

    const std::vector<Vertex> _vertices = {
            { { -.5f, -.5f }, { 1.0f, 0.0f, 0.0f }, { 1.0f, 0.0f } },
            { { 0.5f, -.5f }, { 0.0f, 1.0f, 0.0f }, { 0.0f, 0.0f } },
            { { 0.5f, 0.5f }, { 0.0f, 0.0f, 1.0f }, { 0.0f, 1.0f } },
            { { -.5f, 0.5f }, { 1.0f, 1.0f, 1.0f }, { 1.0f, 1.0f } }
    };
    const std::vector<uint16_t> _indices = { 0, 1, 2, 2, 3, 0 };
    std::array<FrameData, MAX_FRAMES_IN_FLIGHT> _frameData;

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
#if defined(LINUX)
        VK_KHR_MULTIVIEW_EXTENSION_NAME,
        VK_KHR_MAINTENANCE2_EXTENSION_NAME,
#endif
        VK_KHR_SYNCHRONIZATION_2_EXTENSION_NAME,
        VK_KHR_CREATE_RENDERPASS_2_EXTENSION_NAME,
        VK_KHR_DEPTH_STENCIL_RESOLVE_EXTENSION_NAME,
        VK_KHR_DYNAMIC_RENDERING_EXTENSION_NAME,
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
    void CreateDescriptorSetLayout();
    void CreateGraphicsPipeline();
    void CreateCommandPool();
    void CreateCommandBuffers();
    void RecordCommandBuffer(const vk::CommandBuffer& commandBuffer, uint32_t swapChainImageIndex);
    void CreateSyncObjects();
    void CreateBuffer(vk::DeviceSize size, vk::BufferUsageFlags usage, vk::MemoryPropertyFlags properties, vk::Buffer& buffer, vk::DeviceMemory& bufferMemory);
    void CreateVertexBuffer();
    void CreateIndexBuffer();
    void CopyBuffer(vk::Buffer srcBuffer, vk::Buffer dstBuffer, vk::DeviceSize size);
    void CreateUniformBuffers();
    void UpdateUniformData(uint32_t currentFrame);
    uint32_t FindMemoryType(uint32_t typeFilter, vk::MemoryPropertyFlags properties);
    void CreateTextureImage();
    void CreateImage(uint32_t width, uint32_t height, vk::Format format, vk::ImageTiling tiling, vk::ImageUsageFlags usage, vk::MemoryPropertyFlags properties, vk::Image& image, vk::DeviceMemory& memory);
    vk::CommandBuffer BeginSingleTimeCommands(vk::CommandPool& commandPool);
    void EndSingleTimeCommands(vk::CommandBuffer commandBuffer, vk::Queue& queue, vk::CommandPool& commandPool);
    void TransitionImageLayout(vk::Image image, vk::Format format, vk::ImageLayout oldLayout, vk::ImageLayout newLayout);
    void CopyBufferToImage(vk::Buffer buffer, vk::Image image, uint32_t width, uint32_t height);
    vk::ImageView CreateImageView(vk::Image image, vk::Format format);
    void CreateTextureImageView();
    void CreateTextureSampler();

    void LogInstanceExtensions();
    void CreateDescriptorPool();
    void CreateDescriptorSets();
};
