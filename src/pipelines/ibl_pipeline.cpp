#include "pipelines/ibl_pipeline.hpp"
#include "vulkan_helper.hpp"
#include "shaders/shader_loader.hpp"

IBLPipeline::IBLPipeline(const VulkanBrain& brain, const TextureHandle& environmentMap) :
    _brain(brain),
    _environmentMap(environmentMap)
{
    _irradianceMap.size = 32;
    _irradianceMap.format = vk::Format::eR16G16B16A16Sfloat;

    vk::ImageCreateInfo imageCreateInfo{};
    imageCreateInfo.imageType = vk::ImageType::e2D;
    imageCreateInfo.extent.width = _irradianceMap.size;
    imageCreateInfo.extent.height = _irradianceMap.size;
    imageCreateInfo.extent.depth = 1;
    imageCreateInfo.mipLevels = 1;
    imageCreateInfo.arrayLayers = 6;
    imageCreateInfo.format = _irradianceMap.format;
    imageCreateInfo.tiling = vk::ImageTiling::eOptimal;
    imageCreateInfo.initialLayout = vk::ImageLayout::eUndefined;
    imageCreateInfo.usage = vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eColorAttachment;
    imageCreateInfo.sharingMode = vk::SharingMode::eExclusive;
    imageCreateInfo.samples = vk::SampleCountFlagBits::e1;
    imageCreateInfo.flags = vk::ImageCreateFlagBits::eCubeCompatible;

    VmaAllocationCreateInfo allocationInfo{};
    allocationInfo.usage = VMA_MEMORY_USAGE_GPU_ONLY;

    util::VK_ASSERT(vmaCreateImage(_brain.vmaAllocator, reinterpret_cast<VkImageCreateInfo*>(&imageCreateInfo), &allocationInfo, reinterpret_cast<VkImage*>(&_irradianceMap.image), &_irradianceMap.allocation, nullptr), "Failed creating image!");
    vmaSetAllocationName(_brain.vmaAllocator, _irradianceMap.allocation, "Irradiance map");

    for(size_t i = 0; i < 6; ++i)
    {
        vk::ImageViewCreateInfo imageViewCreateInfo{};
        imageViewCreateInfo.image = _irradianceMap.image;
        imageViewCreateInfo.viewType = vk::ImageViewType::e2D;
        imageViewCreateInfo.format = _irradianceMap.format;
        imageViewCreateInfo.subresourceRange.aspectMask = vk::ImageAspectFlagBits::eColor;
        imageViewCreateInfo.subresourceRange.baseMipLevel = 0;
        imageViewCreateInfo.subresourceRange.levelCount = 1;
        imageViewCreateInfo.subresourceRange.baseArrayLayer = i;
        imageViewCreateInfo.subresourceRange.layerCount = 1;

        util::VK_ASSERT(_brain.device.createImageView(&imageViewCreateInfo, nullptr, &_irradianceMap.faceViews[i]), "Failed creating irradiance map image view!");
    }

    vk::ImageViewCreateInfo imageViewCreateInfo{};
    imageViewCreateInfo.image = _irradianceMap.image;
    imageViewCreateInfo.viewType = vk::ImageViewType::eCube;
    imageViewCreateInfo.format = _irradianceMap.format;
    imageViewCreateInfo.subresourceRange.aspectMask = vk::ImageAspectFlagBits::eColor;
    imageViewCreateInfo.subresourceRange.baseMipLevel = 0;
    imageViewCreateInfo.subresourceRange.levelCount = 1;
    imageViewCreateInfo.subresourceRange.baseArrayLayer = 0;
    imageViewCreateInfo.subresourceRange.layerCount = 6;

    util::VK_ASSERT(_brain.device.createImageView(&imageViewCreateInfo, nullptr, &_irradianceMap.view), "Failed creating irradiance map image view!");

    _irradianceMap.sampler = util::CreateSampler(_brain, vk::Filter::eLinear, vk::Filter::eLinear, vk::SamplerAddressMode::eRepeat, vk::SamplerMipmapMode::eLinear, 0);

    CreateDescriptorSetLayout();
    CreateDescriptorSet();
    CreatePipeline();
}

IBLPipeline::~IBLPipeline()
{
    vmaDestroyImage(_brain.vmaAllocator, _irradianceMap.image, _irradianceMap.allocation);
    _brain.device.destroy(_irradianceMap.view);
    for(const auto& view : _irradianceMap.faceViews)
        _brain.device.destroy(view);

    _brain.device.destroy(_pipelineLayout);
    _brain.device.destroy(_pipeline);
    _brain.device.destroy(_descriptorSetLayout);
}

void IBLPipeline::RecordCommands(vk::CommandBuffer commandBuffer)
{
    util::BeginLabel(commandBuffer, "Irradiance pass", glm::vec3{ 17.0f, 138.0f, 178.0f } / 255.0f, _brain.dldi);

    util::TransitionImageLayout(commandBuffer, _irradianceMap.image, _irradianceMap.format, vk::ImageLayout::eUndefined, vk::ImageLayout::eColorAttachmentOptimal, 6, 0, 1);

    for(size_t i = 0; i < 6; ++i)
    {
        vk::RenderingAttachmentInfoKHR finalColorAttachmentInfo{};
        finalColorAttachmentInfo.imageView = _irradianceMap.faceViews[i];
        finalColorAttachmentInfo.imageLayout = vk::ImageLayout::eAttachmentOptimal;
        finalColorAttachmentInfo.storeOp = vk::AttachmentStoreOp::eStore;
        finalColorAttachmentInfo.loadOp = vk::AttachmentLoadOp::eLoad;

        vk::RenderingInfoKHR renderingInfo{};
        renderingInfo.renderArea.extent = vk::Extent2D{ static_cast<uint32_t>(_irradianceMap.size), static_cast<uint32_t>(_irradianceMap.size) };
        renderingInfo.renderArea.offset = vk::Offset2D{ 0, 0 };
        renderingInfo.colorAttachmentCount = 1;
        renderingInfo.pColorAttachments = &finalColorAttachmentInfo;
        renderingInfo.layerCount = 1;
        renderingInfo.pDepthAttachment = nullptr;
        renderingInfo.pStencilAttachment = nullptr;

        commandBuffer.beginRenderingKHR(&renderingInfo, _brain.dldi);

        commandBuffer.pushConstants(_pipelineLayout, vk::ShaderStageFlagBits::eFragment, 0, sizeof(uint32_t), _faceIndices.data() + i);
        commandBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, _pipeline);
        commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, _pipelineLayout, 0, 1, &_descriptorSet, 0, nullptr);

        vk::Viewport viewport = vk::Viewport{ 0.0f, 0.0f, static_cast<float>(_irradianceMap.size), static_cast<float>(_irradianceMap.size), 0.0f,
                                              1.0f };
        commandBuffer.setViewport(0, 1, &viewport);

        vk::Extent2D extent = vk::Extent2D{static_cast<uint32_t>(_irradianceMap.size), static_cast<uint32_t>(_irradianceMap.size)};
        vk::Rect2D scissor = vk::Rect2D{ vk::Offset2D{ 0, 0 }, extent };
        commandBuffer.setScissor(0, 1, &scissor);

        commandBuffer.draw(3, 1, 0, 0);

        commandBuffer.endRenderingKHR(_brain.dldi);
    }

    util::TransitionImageLayout(commandBuffer, _irradianceMap.image, _irradianceMap.format, vk::ImageLayout::eColorAttachmentOptimal, vk::ImageLayout::eShaderReadOnlyOptimal, 6, 0, 1);

    util::EndLabel(commandBuffer, _brain.dldi);
}

void IBLPipeline::CreatePipeline()
{
    vk::PipelineLayoutCreateInfo pipelineLayoutCreateInfo{};
    std::array<vk::DescriptorSetLayout, 1> layouts = { _descriptorSetLayout };
    pipelineLayoutCreateInfo.setLayoutCount = layouts.size();
    pipelineLayoutCreateInfo.pSetLayouts = layouts.data();

    vk::PushConstantRange pushConstantRange{};
    pushConstantRange.size = sizeof(uint32_t) * 6;
    pushConstantRange.offset = 0;
    pushConstantRange.stageFlags = vk::ShaderStageFlagBits::eFragment;
    pipelineLayoutCreateInfo.pushConstantRangeCount = 1;
    pipelineLayoutCreateInfo.pPushConstantRanges = &pushConstantRange;

    util::VK_ASSERT(_brain.device.createPipelineLayout(&pipelineLayoutCreateInfo, nullptr, &_pipelineLayout),
                    "Failed to create IBL pipeline layout!");

    auto vertByteCode = shader::ReadFile("shaders/irradiance-v.spv");
    auto fragByteCode = shader::ReadFile("shaders/irradiance-f.spv");

    vk::ShaderModule vertModule = shader::CreateShaderModule(vertByteCode, _brain.device);
    vk::ShaderModule fragModule = shader::CreateShaderModule(fragByteCode, _brain.device);

    vk::PipelineShaderStageCreateInfo vertShaderStageCreateInfo{};
    vertShaderStageCreateInfo.stage = vk::ShaderStageFlagBits::eVertex;
    vertShaderStageCreateInfo.module = vertModule;
    vertShaderStageCreateInfo.pName = "main";

    vk::PipelineShaderStageCreateInfo fragShaderStageCreateInfo{};
    fragShaderStageCreateInfo.stage = vk::ShaderStageFlagBits::eFragment;
    fragShaderStageCreateInfo.module = fragModule;
    fragShaderStageCreateInfo.pName = "main";

    vk::PipelineShaderStageCreateInfo shaderStages[] = { vertShaderStageCreateInfo, fragShaderStageCreateInfo };

    vk::PipelineVertexInputStateCreateInfo vertexInputStateCreateInfo{};

    vk::PipelineInputAssemblyStateCreateInfo inputAssemblyStateCreateInfo{};
    inputAssemblyStateCreateInfo.topology = vk::PrimitiveTopology::eTriangleList;
    inputAssemblyStateCreateInfo.primitiveRestartEnable = vk::False;

    std::array<vk::DynamicState, 2> dynamicStates = {
            vk::DynamicState::eViewport,
            vk::DynamicState::eScissor,
    };

    vk::PipelineDynamicStateCreateInfo dynamicStateCreateInfo{};
    dynamicStateCreateInfo.dynamicStateCount = dynamicStates.size();
    dynamicStateCreateInfo.pDynamicStates = dynamicStates.data();

    vk::PipelineViewportStateCreateInfo viewportStateCreateInfo{};
    viewportStateCreateInfo.viewportCount = 1;
    viewportStateCreateInfo.scissorCount = 1;

    vk::PipelineRasterizationStateCreateInfo rasterizationStateCreateInfo{};
    rasterizationStateCreateInfo.depthClampEnable = vk::False;
    rasterizationStateCreateInfo.rasterizerDiscardEnable = vk::False;
    rasterizationStateCreateInfo.polygonMode = vk::PolygonMode::eFill;
    rasterizationStateCreateInfo.lineWidth = 1.0f;
    rasterizationStateCreateInfo.cullMode = vk::CullModeFlagBits::eBack;
    rasterizationStateCreateInfo.frontFace = vk::FrontFace::eClockwise;
    rasterizationStateCreateInfo.depthBiasEnable = vk::False;
    rasterizationStateCreateInfo.depthBiasConstantFactor = 0.0f;
    rasterizationStateCreateInfo.depthBiasClamp = 0.0f;
    rasterizationStateCreateInfo.depthBiasSlopeFactor = 0.0f;

    vk::PipelineMultisampleStateCreateInfo multisampleStateCreateInfo{};
    multisampleStateCreateInfo.sampleShadingEnable = vk::False;
    multisampleStateCreateInfo.rasterizationSamples = vk::SampleCountFlagBits::e1;
    multisampleStateCreateInfo.minSampleShading = 1.0f;
    multisampleStateCreateInfo.pSampleMask = nullptr;
    multisampleStateCreateInfo.alphaToCoverageEnable = vk::False;
    multisampleStateCreateInfo.alphaToOneEnable = vk::False;

    std::array<vk::PipelineColorBlendAttachmentState, 1> colorBlendAttachmentStates{};
    for(auto& blendAttachmentState : colorBlendAttachmentStates)
    {
        blendAttachmentState.blendEnable = vk::False;
        blendAttachmentState.colorWriteMask =
                vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG | vk::ColorComponentFlagBits::eB |
                vk::ColorComponentFlagBits::eA;
    }

    vk::PipelineColorBlendStateCreateInfo colorBlendStateCreateInfo{};
    colorBlendStateCreateInfo.logicOpEnable = vk::False;
    colorBlendStateCreateInfo.attachmentCount = colorBlendAttachmentStates.size();
    colorBlendStateCreateInfo.pAttachments = colorBlendAttachmentStates.data();

    vk::PipelineDepthStencilStateCreateInfo depthStencilStateCreateInfo{};
    depthStencilStateCreateInfo.depthTestEnable = true;
    depthStencilStateCreateInfo.depthWriteEnable = true;
    depthStencilStateCreateInfo.depthCompareOp = vk::CompareOp::eLess;
    depthStencilStateCreateInfo.depthBoundsTestEnable = false;
    depthStencilStateCreateInfo.minDepthBounds = 0.0f;
    depthStencilStateCreateInfo.maxDepthBounds = 1.0f;
    depthStencilStateCreateInfo.stencilTestEnable = false;

    vk::GraphicsPipelineCreateInfo pipelineCreateInfo{};
    pipelineCreateInfo.stageCount = 2;
    pipelineCreateInfo.pStages = shaderStages;
    pipelineCreateInfo.pVertexInputState = &vertexInputStateCreateInfo;
    pipelineCreateInfo.pInputAssemblyState = &inputAssemblyStateCreateInfo;
    pipelineCreateInfo.pViewportState = &viewportStateCreateInfo;
    pipelineCreateInfo.pRasterizationState = &rasterizationStateCreateInfo;
    pipelineCreateInfo.pMultisampleState = &multisampleStateCreateInfo;
    pipelineCreateInfo.pDepthStencilState = &depthStencilStateCreateInfo;
    pipelineCreateInfo.pColorBlendState = &colorBlendStateCreateInfo;
    pipelineCreateInfo.pDynamicState = &dynamicStateCreateInfo;
    pipelineCreateInfo.layout = _pipelineLayout;
    pipelineCreateInfo.subpass = 0;
    pipelineCreateInfo.basePipelineHandle = nullptr;
    pipelineCreateInfo.basePipelineIndex = -1;

    vk::PipelineRenderingCreateInfoKHR pipelineRenderingCreateInfoKhr{};
    pipelineRenderingCreateInfoKhr.colorAttachmentCount = 1;
    pipelineRenderingCreateInfoKhr.pColorAttachmentFormats = &_irradianceMap.format;

    pipelineCreateInfo.pNext = &pipelineRenderingCreateInfoKhr;
    pipelineCreateInfo.renderPass = nullptr; // Using dynamic rendering.

    auto result = _brain.device.createGraphicsPipeline(nullptr, pipelineCreateInfo, nullptr);
    util::VK_ASSERT(result.result, "Failed creating the IBL pipeline!");
    _pipeline = result.value;

    _brain.device.destroy(vertModule);
    _brain.device.destroy(fragModule);
}

void IBLPipeline::CreateDescriptorSetLayout()
{
    std::array<vk::DescriptorSetLayoutBinding, 1> bindings{};

    vk::DescriptorSetLayoutBinding& samplerLayoutBinding{bindings[0]};
    samplerLayoutBinding.binding = 0;
    samplerLayoutBinding.descriptorCount = 1;
    samplerLayoutBinding.descriptorType = vk::DescriptorType::eCombinedImageSampler;
    samplerLayoutBinding.stageFlags = vk::ShaderStageFlagBits::eFragment;
    samplerLayoutBinding.pImmutableSamplers = nullptr;

    vk::DescriptorSetLayoutCreateInfo createInfo{};
    createInfo.bindingCount = bindings.size();
    createInfo.pBindings = bindings.data();

    util::VK_ASSERT(_brain.device.createDescriptorSetLayout(&createInfo, nullptr, &_descriptorSetLayout),
                    "Failed creating IBL descriptor set layout!");
}

void IBLPipeline::CreateDescriptorSet()
{
    vk::DescriptorSetAllocateInfo allocateInfo{};
    allocateInfo.descriptorPool = _brain.descriptorPool;
    allocateInfo.descriptorSetCount = 1;
    allocateInfo.pSetLayouts = &_descriptorSetLayout;

    util::VK_ASSERT(_brain.device.allocateDescriptorSets(&allocateInfo, &_descriptorSet),
                    "Failed allocating descriptor sets!");

    vk::DescriptorImageInfo imageInfo{};
    imageInfo.sampler = *_irradianceMap.sampler;
    imageInfo.imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal;
    imageInfo.imageView = _environmentMap.imageView;

    vk::WriteDescriptorSet descriptorWrite{};
    descriptorWrite.dstSet = _descriptorSet;
    descriptorWrite.dstBinding = 0;
    descriptorWrite.dstArrayElement = 0;
    descriptorWrite.descriptorType = vk::DescriptorType::eCombinedImageSampler;
    descriptorWrite.descriptorCount = 1;
    descriptorWrite.pImageInfo = &imageInfo;

    _brain.device.updateDescriptorSets(1, &descriptorWrite, 0, nullptr);
}
