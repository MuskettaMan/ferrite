#include "pipelines/lighting_pipeline.hpp"
#include "shaders/shader_loader.hpp"

LightingPipeline::LightingPipeline(const VulkanBrain& brain, const GBuffers& gBuffers, const HDRTarget& hdrTarget, const CameraStructure& camera) :
    _brain(brain),
    _gBuffers(gBuffers),
    _hdrTarget(hdrTarget),
    _camera(camera)
{
    _sampler = util::CreateSampler(_brain, vk::Filter::eLinear, vk::Filter::eLinear, vk::SamplerAddressMode::eRepeat, vk::SamplerMipmapMode::eLinear);
    CreateDescriptorSetLayout();
    CreateDescriptorSets();
    CreatePipeline();
}

void LightingPipeline::RecordCommands(vk::CommandBuffer commandBuffer, uint32_t currentFrame)
{
    vk::RenderingAttachmentInfoKHR finalColorAttachmentInfo{};
    finalColorAttachmentInfo.imageView = _hdrTarget.imageViews;
    finalColorAttachmentInfo.imageLayout = vk::ImageLayout::eAttachmentOptimalKHR;
    finalColorAttachmentInfo.storeOp = vk::AttachmentStoreOp::eStore;
    finalColorAttachmentInfo.loadOp = vk::AttachmentLoadOp::eLoad;
    finalColorAttachmentInfo.clearValue.color = vk::ClearColorValue{ 0.0f, 0.0f, 0.0f, 0.0f };

    vk::RenderingInfoKHR renderingInfo{};
    renderingInfo.renderArea.extent = vk::Extent2D{ _gBuffers.Size().x, _gBuffers.Size().y };
    renderingInfo.renderArea.offset = vk::Offset2D{ 0, 0 };
    renderingInfo.colorAttachmentCount = 1;
    renderingInfo.pColorAttachments = &finalColorAttachmentInfo;
    renderingInfo.layerCount = 1;
    renderingInfo.pDepthAttachment = nullptr;
    renderingInfo.pStencilAttachment = nullptr;

    util::BeginLabel(commandBuffer, "Lighting pass", glm::vec3{ 255.0f, 209.0f, 102.0f } / 255.0f, _brain.dldi);
    commandBuffer.beginRenderingKHR(&renderingInfo, _brain.dldi);

    commandBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, _pipeline);

    commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, _pipelineLayout, 0, 1, &_descriptorSet, 0, nullptr);
    commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, _pipelineLayout, 1, 1, &_camera.descriptorSets[currentFrame], 0, nullptr);

    // Fullscreen triangle.
    commandBuffer.draw(3, 1, 0, 0);

    commandBuffer.endRenderingKHR(_brain.dldi);
    util::EndLabel(commandBuffer, _brain.dldi);
}

LightingPipeline::~LightingPipeline()
{
    _brain.device.destroy(_pipeline);
    _brain.device.destroy(_pipelineLayout);

    _brain.device.destroy(_descriptorSetLayout);
}

void LightingPipeline::CreatePipeline()
{
    std::array<vk::DescriptorSetLayout, 2> descriptorLayouts = { _descriptorSetLayout, _camera.descriptorSetLayout };

    vk::PipelineLayoutCreateInfo pipelineLayoutCreateInfo{};
    pipelineLayoutCreateInfo.setLayoutCount = descriptorLayouts.size();
    pipelineLayoutCreateInfo.pSetLayouts = descriptorLayouts.data();
    pipelineLayoutCreateInfo.pushConstantRangeCount = 0;
    pipelineLayoutCreateInfo.pPushConstantRanges = nullptr;

    util::VK_ASSERT(_brain.device.createPipelineLayout(&pipelineLayoutCreateInfo, nullptr, &_pipelineLayout),
                    "Failed creating geometry pipeline layout!");

    auto vertByteCode = shader::ReadFile("shaders/lighting-v.spv");
    auto fragByteCode = shader::ReadFile("shaders/lighting-f.spv");

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

    vk::PipelineColorBlendAttachmentState colorBlendAttachmentState{};
    colorBlendAttachmentState.blendEnable = vk::False;
    colorBlendAttachmentState.colorWriteMask =
            vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG | vk::ColorComponentFlagBits::eB |
            vk::ColorComponentFlagBits::eA;

    vk::PipelineColorBlendStateCreateInfo colorBlendStateCreateInfo{};
    colorBlendStateCreateInfo.logicOpEnable = vk::False;
    colorBlendStateCreateInfo.attachmentCount = 1;
    colorBlendStateCreateInfo.pAttachments = &colorBlendAttachmentState;

    vk::PipelineDepthStencilStateCreateInfo depthStencilStateCreateInfo{};
    depthStencilStateCreateInfo.depthTestEnable = false;
    depthStencilStateCreateInfo.depthWriteEnable = false;

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
    pipelineRenderingCreateInfoKhr.pColorAttachmentFormats = &_hdrTarget.format;

    pipelineCreateInfo.pNext = &pipelineRenderingCreateInfoKhr;
    pipelineCreateInfo.renderPass = nullptr; // Using dynamic rendering.

    auto result = _brain.device.createGraphicsPipeline(nullptr, pipelineCreateInfo, nullptr);
    util::VK_ASSERT(result.result, "Failed creating the geometry pipeline layout!");
    _pipeline = result.value;

    _brain.device.destroy(vertModule);
    _brain.device.destroy(fragModule);
}

void LightingPipeline::CreateDescriptorSetLayout()
{
    std::array<vk::DescriptorSetLayoutBinding, DEFERRED_ATTACHMENT_COUNT + 1> bindings{};

    vk::DescriptorSetLayoutBinding& samplerLayoutBinding{bindings[0]};
    samplerLayoutBinding.binding = 0;
    samplerLayoutBinding.descriptorCount = 1;
    samplerLayoutBinding.descriptorType = vk::DescriptorType::eSampler;
    samplerLayoutBinding.stageFlags = vk::ShaderStageFlagBits::eFragment;
    samplerLayoutBinding.pImmutableSamplers = nullptr;

    for(size_t i = 1; i < bindings.size(); ++i)
    {
        vk::DescriptorSetLayoutBinding& binding{bindings[i]};
        binding.binding = i;
        binding.descriptorCount = 1;
        binding.descriptorType = vk::DescriptorType::eSampledImage;
        binding.stageFlags = vk::ShaderStageFlagBits::eFragment;
        binding.pImmutableSamplers = nullptr;
    }

    vk::DescriptorSetLayoutCreateInfo createInfo{};
    createInfo.bindingCount = bindings.size();
    createInfo.pBindings = bindings.data();

    util::VK_ASSERT(_brain.device.createDescriptorSetLayout(&createInfo, nullptr, &_descriptorSetLayout),
                    "Failed creating lighting descriptor set layout!");
}

void LightingPipeline::CreateDescriptorSets()
{
    vk::DescriptorSetAllocateInfo allocateInfo{};
    allocateInfo.descriptorPool = _brain.descriptorPool;
    allocateInfo.descriptorSetCount = 1;
    allocateInfo.pSetLayouts = &_descriptorSetLayout;

    util::VK_ASSERT(_brain.device.allocateDescriptorSets(&allocateInfo, &_descriptorSet),
                    "Failed allocating descriptor sets!");

    UpdateGBufferViews();
}

void LightingPipeline::UpdateGBufferViews()
{
    vk::DescriptorImageInfo samplerInfo{};
    samplerInfo.sampler = *_sampler;

    std::array<vk::DescriptorImageInfo, DEFERRED_ATTACHMENT_COUNT> imageInfo{};
    for(size_t i = 0; i < imageInfo.size(); ++i)
    {
        imageInfo[i].imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal;
        imageInfo[i].imageView = _gBuffers.GBufferView(i);
    }

    std::array<vk::WriteDescriptorSet, DEFERRED_ATTACHMENT_COUNT + 1> descriptorWrites{};
    descriptorWrites[0].dstSet = _descriptorSet;
    descriptorWrites[0].dstBinding = 0;
    descriptorWrites[0].dstArrayElement = 0;
    descriptorWrites[0].descriptorType = vk::DescriptorType::eSampler;
    descriptorWrites[0].descriptorCount = 1;
    descriptorWrites[0].pImageInfo = &samplerInfo;

    for(size_t i = 1; i < descriptorWrites.size(); ++i)
    {
        descriptorWrites[i].dstSet = _descriptorSet;
        descriptorWrites[i].dstBinding = i;
        descriptorWrites[i].dstArrayElement = 0;
        descriptorWrites[i].descriptorType = vk::DescriptorType::eSampledImage;
        descriptorWrites[i].descriptorCount = 1;
        descriptorWrites[i].pImageInfo = &imageInfo[i - 1];
    }

    _brain.device.updateDescriptorSets(descriptorWrites.size(), descriptorWrites.data(), 0, nullptr);
}
