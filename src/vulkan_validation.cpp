#include "vulkan_validation.hpp"
#include <string>
#include <iostream>

vk::Result util::CreateDebugUtilsMessengerEXT(vk::Instance instance, const vk::DebugUtilsMessengerCreateInfoEXT* pCreateInfo,
                                              const vk::AllocationCallbacks* pAllocator, vk::DebugUtilsMessengerEXT* pDebugMessenger)
{
    vk::DispatchLoaderDynamic dldi{ instance, vkGetInstanceProcAddr };
    return instance.createDebugUtilsMessengerEXT(pCreateInfo, pAllocator, pDebugMessenger, dldi);
}

void util::DestroyDebugUtilsMessengerEXT(vk::Instance instance, vk::DebugUtilsMessengerEXT debugMessenger,
                                           const vk::AllocationCallbacks* pAllocator)
{
    vk::DispatchLoaderDynamic dldi{ instance, vkGetInstanceProcAddr };
    return instance.destroyDebugUtilsMessengerEXT(debugMessenger, pAllocator, dldi);
}

void util::PopulateDebugMessengerCreateInfo(vk::DebugUtilsMessengerCreateInfoEXT& createInfo)
{
    createInfo = vk::DebugUtilsMessengerCreateInfoEXT{};
    createInfo.messageSeverity =
            vk::DebugUtilsMessageSeverityFlagBitsEXT::eVerbose |
            vk::DebugUtilsMessageSeverityFlagBitsEXT::eInfo |
            vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning |
            vk::DebugUtilsMessageSeverityFlagBitsEXT::eError;
    createInfo.messageType =
            vk::DebugUtilsMessageTypeFlagBitsEXT::eGeneral |
            vk::DebugUtilsMessageTypeFlagBitsEXT::eValidation |
            vk::DebugUtilsMessageTypeFlagBitsEXT::ePerformance;
    createInfo.pfnUserCallback = DebugCallback;
}

VKAPI_ATTR VkBool32 VKAPI_CALL util::DebugCallback(
        VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
        VkDebugUtilsMessageTypeFlagsEXT messageType,
        const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData,
        void* pUserData)
{
    static std::string type;
    switch(messageType)
    {
    case VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT:
        type = "[GENERAL]";
        break;
    case VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT:
        type = "[VALIDATION]";
        break;
    case VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT:
        type = "[PERFORMANCE]";
        break;
    }

    static std::string severity{};
    switch(messageSeverity)
    {
    case VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT:
        severity = "[VERBOSE]";
        break;
    case VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT:
        severity = "[INFO]";
        break;
    case VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT:
        severity = "[WARNING]";
        break;
    case VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT:
        severity = "[ERROR]";
        break;
    }

    if(messageSeverity >= VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT)
        std::cerr << severity.c_str() << " " << type.c_str() << " Validation layer: " << pCallbackData->pMessage << std::endl;

    return VK_FALSE;
}