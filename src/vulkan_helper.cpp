#include "vulkan_helper.hpp"

void util::VK_ASSERT(vk::Result result, std::string_view message)
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

void util::VK_ASSERT(VkResult result, std::string_view message)
{
    VK_ASSERT(vk::Result(result), message);
}

bool util::HasStencilComponent(vk::Format format)
{
    return format == vk::Format::eD32SfloatS8Uint ||
           format == vk::Format::eD24UnormS8Uint;
}

std::optional<vk::Format> util::FindSupportedFormat(const vk::PhysicalDevice physicalDevice, const std::vector<vk::Format>& candidates, vk::ImageTiling tiling,
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

vk::ImageView util::CreateImageView(vk::Device device, vk::Image image, vk::Format format, vk::ImageAspectFlags aspectFlags, uint32_t layer, uint32_t mipCount)
{
    vk::ImageViewCreateInfo createInfo{};
    createInfo.image = image;
    createInfo.viewType = vk::ImageViewType::e2D;
    createInfo.format = format;
    createInfo.subresourceRange.aspectMask = aspectFlags;
    createInfo.subresourceRange.baseMipLevel = 0;
    createInfo.subresourceRange.levelCount = mipCount;
    createInfo.subresourceRange.baseArrayLayer = layer;
    createInfo.subresourceRange.layerCount = 1;

    vk::ImageView view;
    util::VK_ASSERT(device.createImageView(&createInfo, nullptr, &view), "Failed creating image view!");

    return view;
}

uint32_t util::FindMemoryType(vk::PhysicalDevice physicalDevice, uint32_t typeFilter, vk::MemoryPropertyFlags properties)
{
    vk::PhysicalDeviceMemoryProperties memoryProperties;
    physicalDevice.getMemoryProperties(&memoryProperties);

    for(uint32_t i = 0; i < memoryProperties.memoryTypeCount; ++i)
        if(typeFilter & (1 << i) && (memoryProperties.memoryTypes[i].propertyFlags & properties) == properties)
            return i;

    throw std::runtime_error("Failed finding suitable memory type!");
}

void util::CreateImage(VmaAllocator allocator, uint32_t width, uint32_t height, vk::Format format, vk::ImageTiling tiling, vk::ImageUsageFlags usage, vk::Image& image, VmaAllocation& allocation, std::string_view name, bool generateMips, VmaMemoryUsage memoryUsage, uint32_t numLayers)
{
    uint32_t mipCount = 1;
    if(generateMips)
        mipCount = static_cast<uint32_t>(floor(log2(std::max(width, height))) + 1);

    vk::ImageCreateInfo createInfo{};
    createInfo.imageType = vk::ImageType::e2D;
    createInfo.extent.width = width;
    createInfo.extent.height = height;
    createInfo.extent.depth = 1;
    createInfo.mipLevels = mipCount;
    createInfo.arrayLayers = numLayers;
    createInfo.format = format;
    createInfo.tiling = tiling;
    createInfo.initialLayout = vk::ImageLayout::eUndefined;
    createInfo.usage = usage;
    createInfo.sharingMode = vk::SharingMode::eExclusive;
    createInfo.samples = vk::SampleCountFlagBits::e1;
    createInfo.flags = vk::ImageCreateFlags{ 0 };

    VmaAllocationCreateInfo allocationInfo{};
    allocationInfo.usage = memoryUsage;

    util::VK_ASSERT(vmaCreateImage(allocator, reinterpret_cast<VkImageCreateInfo*>(&createInfo), &allocationInfo, reinterpret_cast<VkImage*>(&image), &allocation, nullptr), "Failed creating image!");
    vmaSetAllocationName(allocator, allocation, name.data());
}

void util::CreateBuffer(const VulkanBrain& brain, vk::DeviceSize size, vk::BufferUsageFlags usage, vk::Buffer& buffer, bool mappable, VmaAllocation& allocation, VmaMemoryUsage memoryUsage, std::string_view name)
{
    vk::BufferCreateInfo bufferInfo{};
    bufferInfo.size = size;
    bufferInfo.usage = usage;
    bufferInfo.sharingMode = vk::SharingMode::eExclusive;
    bufferInfo.queueFamilyIndexCount = 1;
    bufferInfo.pQueueFamilyIndices = &brain.queueFamilyIndices.graphicsFamily.value();

    VmaAllocationCreateInfo allocationInfo{};
    allocationInfo.usage = memoryUsage;
    if(mappable)
        allocationInfo.flags |= VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT;

    util::VK_ASSERT(vmaCreateBuffer(brain.vmaAllocator, reinterpret_cast<VkBufferCreateInfo*>(&bufferInfo), &allocationInfo, reinterpret_cast<VkBuffer*>(&buffer), &allocation, nullptr), "Failed creating buffer!");
    vmaSetAllocationName(brain.vmaAllocator, allocation, name.data());
}

vk::CommandBuffer util::BeginSingleTimeCommands(const VulkanBrain& brain)
{
    vk::CommandBufferAllocateInfo allocateInfo{};
    allocateInfo.level = vk::CommandBufferLevel::ePrimary;
    allocateInfo.commandPool = brain.commandPool;
    allocateInfo.commandBufferCount = 1;

    vk::CommandBuffer commandBuffer;
    util::VK_ASSERT(brain.device.allocateCommandBuffers(&allocateInfo, &commandBuffer), "Failed allocating one time command buffer!");

    vk::CommandBufferBeginInfo beginInfo{};
    beginInfo.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit;

    util::VK_ASSERT(commandBuffer.begin(&beginInfo), "Failed beginning one time command buffer!");

    return commandBuffer;
}

void util::EndSingleTimeCommands(const VulkanBrain& brain, vk::CommandBuffer commandBuffer)
{
    commandBuffer.end();

    vk::SubmitInfo submitInfo{};
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &commandBuffer;

    util::VK_ASSERT(brain.graphicsQueue.submit(1, &submitInfo, nullptr), "Failed submitting one time buffer to queue!");
    brain.graphicsQueue.waitIdle();

    brain.device.free(brain.commandPool, commandBuffer);
}

void util::CopyBuffer(vk::CommandBuffer commandBuffer, vk::Buffer srcBuffer, vk::Buffer dstBuffer, vk::DeviceSize size)
{
    vk::BufferCopy copyRegion{};
    copyRegion.srcOffset = 0;
    copyRegion.dstOffset = 0;
    copyRegion.size = size;
    commandBuffer.copyBuffer(srcBuffer, dstBuffer, 1, &copyRegion);
}

MaterialHandle util::CreateMaterial(const VulkanBrain& brain, const std::array<std::shared_ptr<TextureHandle>, 5>& textures, const MaterialHandle::MaterialInfo& info, vk::Sampler sampler, vk::DescriptorSetLayout materialLayout, std::shared_ptr<MaterialHandle> defaultMaterial)
{
    MaterialHandle materialHandle;
    materialHandle.textures = textures;

    util::CreateBuffer(brain, sizeof(MaterialHandle::MaterialInfo), vk::BufferUsageFlagBits::eUniformBuffer, materialHandle.materialUniformBuffer, true, materialHandle.materialUniformAllocation, VMA_MEMORY_USAGE_CPU_ONLY, "Material uniform buffer");

    void* uniformPtr;
    util::VK_ASSERT(vmaMapMemory(brain.vmaAllocator, materialHandle.materialUniformAllocation, &uniformPtr), "Failed mapping memory for material UBO!");
    std::memcpy(uniformPtr, &info, sizeof(info));
    vmaUnmapMemory(brain.vmaAllocator, materialHandle.materialUniformAllocation);


    vk::DescriptorSetAllocateInfo allocateInfo{};
    allocateInfo.descriptorPool = brain.descriptorPool;
    allocateInfo.descriptorSetCount = 1;
    allocateInfo.pSetLayouts = &materialLayout;

    util::VK_ASSERT(brain.device.allocateDescriptorSets(&allocateInfo, &materialHandle.descriptorSet),
                    "Failed allocating material descriptor set!");

    std::array<vk::DescriptorImageInfo, 6> imageInfos;
    imageInfos[0].sampler = sampler;
    for(size_t i = 1; i < MaterialHandle::TEXTURE_COUNT + 1; ++i)
    {
        const MaterialHandle& material = textures[i - 1] != nullptr ? materialHandle : *defaultMaterial;

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

    brain.device.updateDescriptorSets(writes.size(), writes.data(), 0, nullptr);

    return materialHandle;
}

vk::UniqueSampler util::CreateSampler(const VulkanBrain& brain, vk::Filter min, vk::Filter mag, vk::SamplerAddressMode addressingMode, vk::SamplerMipmapMode mipmapMode, uint32_t mipLevels)
{
    vk::PhysicalDeviceProperties properties{};
    brain.physicalDevice.getProperties(&properties);

    vk::SamplerCreateInfo createInfo{};
    createInfo.magFilter = mag;
    createInfo.minFilter = min;
    createInfo.addressModeU = addressingMode;
    createInfo.addressModeV = addressingMode;
    createInfo.addressModeW = addressingMode;
    createInfo.anisotropyEnable = 1;
    createInfo.maxAnisotropy = properties.limits.maxSamplerAnisotropy;
    createInfo.borderColor = vk::BorderColor::eIntOpaqueBlack;
    createInfo.unnormalizedCoordinates = 0;
    createInfo.compareEnable = 0;
    createInfo.compareOp = vk::CompareOp::eAlways;
    createInfo.mipmapMode = mipmapMode;
    createInfo.mipLodBias = 0.0f;
    createInfo.minLod = 0.0f;
    createInfo.maxLod = static_cast<float>(mipLevels);

    return brain.device.createSamplerUnique(createInfo);
}

void util::TransitionImageLayout(vk::CommandBuffer commandBuffer, vk::Image image, vk::Format format, vk::ImageLayout oldLayout, vk::ImageLayout newLayout, uint32_t numLayers, uint32_t mipLevel, uint32_t mipCount)
{
    vk::ImageMemoryBarrier barrier{};
    barrier.oldLayout = oldLayout;
    barrier.newLayout = newLayout;
    barrier.srcQueueFamilyIndex = vk::QueueFamilyIgnored;
    barrier.dstQueueFamilyIndex = vk::QueueFamilyIgnored;
    barrier.image = image;
    barrier.subresourceRange.aspectMask = vk::ImageAspectFlagBits::eColor;
    barrier.subresourceRange.baseMipLevel = mipLevel;
    barrier.subresourceRange.levelCount = mipCount;
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
    else if(oldLayout == vk::ImageLayout::eTransferDstOptimal && newLayout == vk::ImageLayout::eTransferSrcOptimal)
    {
        barrier.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
        barrier.dstAccessMask = vk::AccessFlagBits::eTransferRead;

        sourceStage = vk::PipelineStageFlagBits::eTransfer;
        destinationStage = vk::PipelineStageFlagBits::eTransfer;
    }
    else if (oldLayout == vk::ImageLayout::eTransferDstOptimal && newLayout == vk::ImageLayout::eShaderReadOnlyOptimal)
    {
        barrier.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
        barrier.dstAccessMask = vk::AccessFlagBits::eShaderRead;

        sourceStage = vk::PipelineStageFlagBits::eTransfer;
        destinationStage = vk::PipelineStageFlagBits::eFragmentShader;
    }
    else if (oldLayout == vk::ImageLayout::eTransferSrcOptimal && newLayout == vk::ImageLayout::eShaderReadOnlyOptimal)
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

void util::CopyBufferToImage(vk::CommandBuffer commandBuffer, vk::Buffer buffer, vk::Image image, uint32_t width, uint32_t height)
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

    commandBuffer.copyBufferToImage(buffer, image, vk::ImageLayout::eTransferDstOptimal, 1, &region);
}

void util::BeginLabel(vk::Queue queue, std::string_view label, glm::vec3 color, const vk::DispatchLoaderDynamic dldi)
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

void util::EndLabel(vk::Queue queue, const vk::DispatchLoaderDynamic dldi)
{
#if defined(NDEBUG)
    return;
#endif
    queue.endDebugUtilsLabelEXT(dldi);
}

void util::BeginLabel(vk::CommandBuffer commandBuffer, std::string_view label, glm::vec3 color, const vk::DispatchLoaderDynamic dldi)
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

void util::EndLabel(vk::CommandBuffer commandBuffer, const vk::DispatchLoaderDynamic dldi)
{
#if defined(NDEBUG)
    return;
#endif
    commandBuffer.endDebugUtilsLabelEXT(dldi);
}

