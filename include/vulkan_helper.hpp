#pragma once

#include <vulkan/vulkan.hpp>
#include <magic_enum.hpp>
#include <string>
#include <glm/glm.hpp>
#include <sstream>
#include "spdlog/spdlog.h"
#include "vk_mem_alloc.h"

namespace util
{
    static void VK_ASSERT(vk::Result result, std::string_view message)
    {
        if(result == vk::Result::eSuccess)
            return;

        static std::string completeMessage{};
        completeMessage = "[] ";
        auto resultStr = magic_enum::enum_name(result);

        completeMessage.insert(0, resultStr);
        completeMessage.insert(completeMessage.size() - 1, message);

        throw std::runtime_error(completeMessage.c_str());
    }

    static void VK_ASSERT(VkResult result, std::string_view message)
    {
        VK_ASSERT(vk::Result(result), message);
    }

    static bool HasStencilComponent(vk::Format format)
    {
        return format == vk::Format::eD32SfloatS8Uint ||
               format == vk::Format::eD24UnormS8Uint;
    }

    static std::optional<vk::Format> FindSupportedFormat(const vk::PhysicalDevice physicalDevice, const std::vector<vk::Format>& candidates, vk::ImageTiling tiling,
                                                          vk::FormatFeatureFlags features)
    {
        for(vk::Format format : candidates)
        {
            vk::FormatProperties props;
            physicalDevice.getFormatProperties(format, &props);

            if(tiling == vk::ImageTiling::eLinear && (props.linearTilingFeatures & features) == features)
                return format;
            else if(tiling == vk::ImageTiling::eOptimal && (props.optimalTilingFeatures & features) == features)
                return format;
        }

        return std::nullopt;
    }

    static vk::ImageView CreateImageView(vk::Device device, vk::Image image, vk::Format format, vk::ImageAspectFlags aspectFlags, uint32_t layer = 0)
    {
        vk::ImageViewCreateInfo createInfo{};
        createInfo.image = image;
        createInfo.viewType = vk::ImageViewType::e2D;
        createInfo.format = format;
        createInfo.subresourceRange.aspectMask = aspectFlags;
        createInfo.subresourceRange.baseMipLevel = 0;
        createInfo.subresourceRange.levelCount = 1;
        createInfo.subresourceRange.baseArrayLayer = layer;
        createInfo.subresourceRange.layerCount = 1;

        vk::ImageView view;
        util::VK_ASSERT(device.createImageView(&createInfo, nullptr, &view), "Failed creating image view!");

        return view;
    }

    static uint32_t FindMemoryType(vk::PhysicalDevice physicalDevice, uint32_t typeFilter, vk::MemoryPropertyFlags properties)
    {
        vk::PhysicalDeviceMemoryProperties memoryProperties;
        physicalDevice.getMemoryProperties(&memoryProperties);

        for(uint32_t i = 0; i < memoryProperties.memoryTypeCount; ++i)
            if(typeFilter & (1 << i) && (memoryProperties.memoryTypes[i].propertyFlags & properties) == properties)
                return i;

        throw std::runtime_error("Failed finding suitable memory type!");
    }

    static void CreateImage(VmaAllocator allocator, uint32_t width, uint32_t height, vk::Format format, vk::ImageTiling tiling, vk::ImageUsageFlags usage, vk::Image& image, VmaAllocation& allocation, uint32_t numLayers = 1)
    {
        vk::ImageCreateInfo createInfo{};
        createInfo.imageType = vk::ImageType::e2D;
        createInfo.extent.width = width;
        createInfo.extent.height = height;
        createInfo.extent.depth = 1;
        createInfo.mipLevels = 1;
        createInfo.arrayLayers = numLayers;
        createInfo.format = format;
        createInfo.tiling = tiling;
        createInfo.initialLayout = vk::ImageLayout::eUndefined;
        createInfo.usage = usage;
        createInfo.sharingMode = vk::SharingMode::eExclusive;
        createInfo.samples = vk::SampleCountFlagBits::e1;
        createInfo.flags = vk::ImageCreateFlags{ 0 };

        VmaAllocationCreateInfo allocationInfo{};
        allocationInfo.usage = VMA_MEMORY_USAGE_AUTO;

        util::VK_ASSERT(vmaCreateImage(allocator, reinterpret_cast<VkImageCreateInfo*>(&createInfo), &allocationInfo, reinterpret_cast<VkImage*>(&image), &allocation, nullptr), "Failed creating image!");
    }

    static vk::CommandBuffer BeginSingleTimeCommands(vk::Device device, vk::CommandPool commandPool)
    {
        vk::CommandBufferAllocateInfo allocateInfo{};
        allocateInfo.level = vk::CommandBufferLevel::ePrimary;
        allocateInfo.commandPool = commandPool;
        allocateInfo.commandBufferCount = 1;

        vk::CommandBuffer commandBuffer;
        util::VK_ASSERT(device.allocateCommandBuffers(&allocateInfo, &commandBuffer), "Failed allocating one time command buffer!");

        vk::CommandBufferBeginInfo beginInfo{};
        beginInfo.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit;

        util::VK_ASSERT(commandBuffer.begin(&beginInfo), "Failed beginning one time command buffer!");

        return commandBuffer;
    }

    static void EndSingleTimeCommands(vk::Device device, vk::Queue queue, vk::CommandBuffer commandBuffer, vk::CommandPool commandPool)
    {
        commandBuffer.end();

        vk::SubmitInfo submitInfo{};
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &commandBuffer;

        util::VK_ASSERT(queue.submit(1, &submitInfo, nullptr), "Failed submitting one time buffer to queue!");
        queue.waitIdle();

        device.free(commandPool, commandBuffer);
    }

    static void TransitionImageLayout(vk::CommandBuffer commandBuffer, vk::Image image, vk::Format format, vk::ImageLayout oldLayout, vk::ImageLayout newLayout, uint32_t numLayers = 1)
    {
        vk::ImageMemoryBarrier barrier{};
        barrier.oldLayout = oldLayout;
        barrier.newLayout = newLayout;
        barrier.srcQueueFamilyIndex = vk::QueueFamilyIgnored;
        barrier.dstQueueFamilyIndex = vk::QueueFamilyIgnored;
        barrier.image = image;
        barrier.subresourceRange.aspectMask = vk::ImageAspectFlagBits::eColor;
        barrier.subresourceRange.baseMipLevel = 0;
        barrier.subresourceRange.levelCount = 1;
        barrier.subresourceRange.baseArrayLayer = 0;
        barrier.subresourceRange.layerCount = numLayers;

        vk::PipelineStageFlags sourceStage;
        vk::PipelineStageFlags destinationStage;

        if(newLayout == vk::ImageLayout::eDepthStencilAttachmentOptimal)
        {
            barrier.subresourceRange.aspectMask = vk::ImageAspectFlagBits::eDepth;
            if(util::HasStencilComponent(format))
                barrier.subresourceRange.aspectMask |= vk::ImageAspectFlagBits::eStencil;
        }

        if(oldLayout == vk::ImageLayout::eUndefined && newLayout == vk::ImageLayout::eTransferDstOptimal)
        {
            barrier.srcAccessMask = vk::AccessFlags{ 0 };
            barrier.dstAccessMask = vk::AccessFlagBits::eTransferWrite;

            sourceStage = vk::PipelineStageFlagBits::eTopOfPipe;
            destinationStage = vk::PipelineStageFlagBits::eTransfer;
        }
        else if (oldLayout == vk::ImageLayout::eTransferDstOptimal && newLayout == vk::ImageLayout::eShaderReadOnlyOptimal)
        {
            barrier.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
            barrier.dstAccessMask = vk::AccessFlagBits::eShaderRead;

            sourceStage = vk::PipelineStageFlagBits::eTransfer;
            destinationStage = vk::PipelineStageFlagBits::eFragmentShader;
        }
        else if (oldLayout == vk::ImageLayout::eColorAttachmentOptimal && newLayout == vk::ImageLayout::eShaderReadOnlyOptimal)
        {
            barrier.srcAccessMask = vk::AccessFlagBits::eColorAttachmentWrite;
            barrier.dstAccessMask = vk::AccessFlagBits::eShaderRead;

            sourceStage = vk::PipelineStageFlagBits::eColorAttachmentOutput;
            destinationStage = vk::PipelineStageFlagBits::eFragmentShader;
        }
        else if (oldLayout == vk::ImageLayout::eUndefined && newLayout == vk::ImageLayout::eColorAttachmentOptimal)
        {
            sourceStage = vk::PipelineStageFlagBits::eTopOfPipe;
            destinationStage = vk::PipelineStageFlagBits::eColorAttachmentOutput;
        }
        else if (oldLayout == vk::ImageLayout::eColorAttachmentOptimal && newLayout == vk::ImageLayout::ePresentSrcKHR)
        {
            sourceStage = vk::PipelineStageFlagBits::eColorAttachmentOutput;
            destinationStage = vk::PipelineStageFlagBits::eBottomOfPipe;
        }
        else if(oldLayout == vk::ImageLayout::eUndefined && newLayout == vk::ImageLayout::eDepthStencilAttachmentOptimal)
        {
            barrier.srcAccessMask = vk::AccessFlags{ 0 };
            barrier.dstAccessMask = vk::AccessFlagBits::eDepthStencilAttachmentRead | vk::AccessFlagBits::eDepthStencilAttachmentWrite;

            sourceStage = vk::PipelineStageFlagBits::eTopOfPipe;
            destinationStage = vk::PipelineStageFlagBits::eEarlyFragmentTests;
        }
        else
            throw std::runtime_error("Unsupported layout transition!");

        commandBuffer.pipelineBarrier(sourceStage, destinationStage,
                                      vk::DependencyFlags{ 0 },
                                      0, nullptr,
                                      0, nullptr,
                                      1, &barrier);
    }

    static void BeginLabel(vk::Queue queue, std::string_view label, glm::vec3 color, const vk::DispatchLoaderDynamic dldi)
    {
#if defined(NDEBUG)
        return;
#endif
        vk::DebugUtilsLabelEXT labelExt{};
        memcpy(labelExt.color.data(), &color.r, sizeof(glm::vec3));
        labelExt.color[3] = 1.0f;
        labelExt.pLabelName = label.data();

        queue.beginDebugUtilsLabelEXT(&labelExt, dldi);
    }

    static void EndLabel(vk::Queue queue, const vk::DispatchLoaderDynamic dldi)
    {
#if defined(NDEBUG)
        return;
#endif
        queue.endDebugUtilsLabelEXT(dldi);
    }

    static void BeginLabel(vk::CommandBuffer commandBuffer, std::string_view label, glm::vec3 color, const vk::DispatchLoaderDynamic dldi)
    {
#if defined(NDEBUG)
        return;
#endif
        vk::DebugUtilsLabelEXT labelExt{};
        memcpy(labelExt.color.data(), &color.r, sizeof(glm::vec3));
        labelExt.color[3] = 1.0f;
        labelExt.pLabelName = label.data();

        commandBuffer.beginDebugUtilsLabelEXT(&labelExt, dldi);
    }

    static void EndLabel(vk::CommandBuffer commandBuffer, const vk::DispatchLoaderDynamic dldi)
    {
#if defined(NDEBUG)
        return;
#endif
        commandBuffer.endDebugUtilsLabelEXT(dldi);
    }

    template <typename T>
    static void NameObject(T object, std::string_view label, vk::Device device, const vk::DispatchLoaderDynamic dldi)
    {
#if defined(NDEBUG)
        return;
#endif
        vk::DebugUtilsObjectNameInfoEXT nameInfo{};

        nameInfo.pObjectName = label.data();
        nameInfo.objectType = object.objectType;
        nameInfo.objectHandle = reinterpret_cast<uint64_t>(static_cast<typename T::CType>(object));

        vk::Result result = device.setDebugUtilsObjectNameEXT(&nameInfo, dldi);
        if (result != vk::Result::eSuccess)
            spdlog::warn("Failed debug naming object!");
    }
}
