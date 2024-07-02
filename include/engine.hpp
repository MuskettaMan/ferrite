#pragma once

#include <memory>
#include <optional>
#include <functional>
#include <vulkan/vulkan.hpp>
#include "class_decorations.hpp"

class Engine
{
public:
    struct InitInfo
    {
        uint32_t extensionCount{ 0 };
        const char** extensions{ nullptr };
        uint32_t width, height;

        std::function<vk::SurfaceKHR(vk::Instance)> retrieveSurface;
    };
    struct QueueFamilyIndices
    {
        std::optional<uint32_t> graphicsFamily;
        std::optional<uint32_t> presentFamily;

        bool IsComplete()
        {
            return graphicsFamily.has_value() && presentFamily.has_value();
        }
    };
    struct SwapChainSupportDetails
    {
        vk::SurfaceCapabilitiesKHR capabilities;
        std::vector<vk::SurfaceFormatKHR> formats;
        std::vector<vk::PresentModeKHR> presentModes;
    };

    Engine();
    NON_COPYABLE(Engine);
    NON_MOVABLE(Engine);

    void Init(const InitInfo& initInfo);
    void Run();
    void Shutdown();

private:
    vk::Instance _instance;
    vk::PhysicalDevice _physicalDevice;
    vk::Device _device;
    vk::Queue _graphicsQueue;
    vk::Queue _presentQueue;
    vk::SurfaceKHR _surface;

    vk::SwapchainKHR _swapChain;
    std::vector<vk::Image> _swapChainImages;
    std::vector<vk::ImageView> _swapChainImageViews;
    vk::Format _swapChainFormat;
    vk::Extent2D _swapChainExtent;

    vk::DebugUtilsMessengerEXT _debugMessenger;

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
        VK_KHR_SWAPCHAIN_EXTENSION_NAME
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
    SwapChainSupportDetails QuerySwapChainSupport(const vk::PhysicalDevice& device);
    void CreateSwapChain(const InitInfo& initInfo);
    vk::SurfaceFormatKHR ChooseSwapSurfaceFormat(const std::vector<vk::SurfaceFormatKHR>& availableFormats);
    vk::PresentModeKHR ChoosePresentMode(const std::vector<vk::PresentModeKHR>& availablePresentModes);
    vk::Extent2D ChooseSwapExtent(const vk::SurfaceCapabilitiesKHR& capabilities, const InitInfo& initInfo);
    void CreateSwapChainImageViews();

    void LogInstanceExtensions();

};
