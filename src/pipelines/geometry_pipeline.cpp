#include "pipelines/geometry_pipeline.hpp"
#include "shaders/shader_loader.hpp"

VkDeviceSize align(VkDeviceSize value, VkDeviceSize alignment) {
    return (value + alignment - 1) & ~(alignment - 1);
}

GeometryPipeline::GeometryPipeline(const VulkanBrain& brain, const GBuffers& gBuffers, vk::DescriptorSetLayout materialDescriptorSetLayout) :
    _brain(brain),
    _gBuffers(gBuffers)
{
    CreateDescriptorSetLayout();
    CreateUniformBuffers();
    CreateDescriptorSets();
    CreatePipeline(materialDescriptorSetLayout);

    spdlog::info("ubo size: {}, minUniformSize: {}, aligned size: {}", sizeof(UBO), _brain.minUniformBufferOffsetAlignment, align(sizeof(UBO), _brain.minUniformBufferOffsetAlignment));
}

GeometryPipeline::~GeometryPipeline()
{
    _brain.device.destroy(_pipeline);
    _brain.device.destroy(_pipelineLayout);
    for(size_t i = 0; i < _frameData.size(); ++i)
    {
        vmaUnmapMemory(_brain.vmaAllocator, _frameData[i].uniformBufferAllocation);
        vmaDestroyBuffer(_brain.vmaAllocator, _frameData[i].uniformBuffer, _frameData[i].uniformBufferAllocation);
    }
    _brain.device.destroy(_descriptorSetLayout);
}

void GeometryPipeline::RecordCommands(vk::CommandBuffer commandBuffer, uint32_t currentFrame, const ModelHandle& model)
{
    std::array<vk::RenderingAttachmentInfoKHR, DEFERRED_ATTACHMENT_COUNT> colorAttachmentInfos{};
    for(size_t i = 0; i < colorAttachmentInfos.size(); ++i)
    {
        vk::RenderingAttachmentInfoKHR& info{ colorAttachmentInfos[i] };
        info.imageView = _gBuffers.GBufferView(currentFrame, i);
        info.imageLayout = vk::ImageLayout::eColorAttachmentOptimal;
        info.storeOp = vk::AttachmentStoreOp::eStore;
        info.loadOp = vk::AttachmentLoadOp::eClear;
        info.clearValue.color = vk::ClearColorValue{ 0.0f, 0.0f, 0.0f, 0.0f };
    }

    vk::RenderingAttachmentInfoKHR depthAttachmentInfo{};
    depthAttachmentInfo.imageView = _gBuffers.DepthImageView();
    depthAttachmentInfo.imageLayout = vk::ImageLayout::eDepthStencilAttachmentOptimal;
    depthAttachmentInfo.storeOp = vk::AttachmentStoreOp::eDontCare;
    depthAttachmentInfo.loadOp = vk::AttachmentLoadOp::eClear;
    depthAttachmentInfo.clearValue.depthStencil = vk::ClearDepthStencilValue{ 1.0f, 0 };

    vk::RenderingAttachmentInfoKHR  stencilAttachmentInfo{depthAttachmentInfo};
    stencilAttachmentInfo.storeOp = vk::AttachmentStoreOp::eDontCare;
    stencilAttachmentInfo.loadOp = vk::AttachmentLoadOp::eDontCare;
    stencilAttachmentInfo.clearValue.depthStencil = vk::ClearDepthStencilValue{ 1.0f, 0 };

    vk::RenderingInfoKHR renderingInfo{};
    glm::uvec2 displaySize = _gBuffers.Size();
    renderingInfo.renderArea.extent = vk::Extent2D{ displaySize.x, displaySize.y };
    renderingInfo.renderArea.offset = vk::Offset2D{ 0, 0 };
    renderingInfo.colorAttachmentCount = colorAttachmentInfos.size();
    renderingInfo.pColorAttachments = colorAttachmentInfos.data();
    renderingInfo.layerCount = 1;
    renderingInfo.pDepthAttachment = &depthAttachmentInfo;
    renderingInfo.pStencilAttachment = util::HasStencilComponent(_gBuffers.DepthFormat()) ? &stencilAttachmentInfo : nullptr;

    util::BeginLabel(commandBuffer, "Geometry pass", glm::vec3{ 6.0f, 214.0f, 160.0f } / 255.0f, _brain.dldi);

    commandBuffer.beginRenderingKHR(&renderingInfo, _brain.dldi);

    commandBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, _pipeline);

    commandBuffer.setViewport(0, 1, &_gBuffers.Viewport());
    commandBuffer.setScissor(0, 1, &_gBuffers.Scissor());

    std::vector<glm::mat4> transforms;
    transforms.reserve(model.hierarchy.allNodes.size());
    for(auto& node : model.hierarchy.allNodes)
    {
        transforms.emplace_back(node.transform);
    }
    UpdateUniformData(currentFrame, transforms);

    for(size_t i = 0; i < model.hierarchy.allNodes.size(); ++i)
    {
        const auto& node = model.hierarchy.allNodes[i];

        for(const auto& primitive : node.mesh->primitives)
        {
            if(primitive.topology != vk::PrimitiveTopology::eTriangleList)
                throw std::runtime_error("No support for topology other than triangle list!");

            assert(primitive.material && "There should always be a material available.");
            const MaterialHandle& material = *primitive.material;

            uint32_t dynamicOffset = static_cast<uint32_t>(i * sizeof(UBO));

            commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, _pipelineLayout, 0, 1, &_frameData[currentFrame].descriptorSet, 1, &dynamicOffset);
            commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, _pipelineLayout, 1, 1, &material.descriptorSet, 0, nullptr);

            vk::Buffer vertexBuffers[] = { primitive.vertexBuffer };
            vk::DeviceSize offsets[] = { 0 };
            commandBuffer.bindVertexBuffers(0, 1, vertexBuffers, offsets);
            commandBuffer.bindIndexBuffer(primitive.indexBuffer, 0, primitive.indexType);

            commandBuffer.drawIndexed(primitive.indexCount, 1, 0, 0, 0);
        }
    }

    commandBuffer.endRenderingKHR(_brain.dldi);

    util::EndLabel(commandBuffer, _brain.dldi);
}

void GeometryPipeline::CreatePipeline(vk::DescriptorSetLayout materialDescriptorSetLayout)
{
    vk::PipelineLayoutCreateInfo pipelineLayoutCreateInfo{};
    std::array<vk::DescriptorSetLayout, 2> layouts = { _descriptorSetLayout, materialDescriptorSetLayout };
    pipelineLayoutCreateInfo.setLayoutCount = layouts.size();
    pipelineLayoutCreateInfo.pSetLayouts = layouts.data();
    pipelineLayoutCreateInfo.pushConstantRangeCount = 0;
    pipelineLayoutCreateInfo.pPushConstantRanges = nullptr;

    util::VK_ASSERT(_brain.device.createPipelineLayout(&pipelineLayoutCreateInfo, nullptr, &_pipelineLayout),
                    "Failed creating geometry pipeline layout!");

    auto vertByteCode = shader::ReadFile("shaders/geom-v.spv");
    auto fragByteCode = shader::ReadFile("shaders/geom-f.spv");

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

    auto bindingDesc = Vertex::GetBindingDescription();
    auto attributes = Vertex::GetAttributeDescriptions();

    vk::PipelineVertexInputStateCreateInfo vertexInputStateCreateInfo{};
    vertexInputStateCreateInfo.vertexBindingDescriptionCount = 1;
    vertexInputStateCreateInfo.pVertexBindingDescriptions = &bindingDesc;
    vertexInputStateCreateInfo.vertexAttributeDescriptionCount = attributes.size();
    vertexInputStateCreateInfo.pVertexAttributeDescriptions = attributes.data();

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
    rasterizationStateCreateInfo.frontFace = vk::FrontFace::eCounterClockwise;
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

    std::array<vk::PipelineColorBlendAttachmentState, DEFERRED_ATTACHMENT_COUNT> colorBlendAttachmentStates{};
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

    vk::GraphicsPipelineCreateInfo geometryPipelineCreateInfo{};
    geometryPipelineCreateInfo.stageCount = 2;
    geometryPipelineCreateInfo.pStages = shaderStages;
    geometryPipelineCreateInfo.pVertexInputState = &vertexInputStateCreateInfo;
    geometryPipelineCreateInfo.pInputAssemblyState = &inputAssemblyStateCreateInfo;
    geometryPipelineCreateInfo.pViewportState = &viewportStateCreateInfo;
    geometryPipelineCreateInfo.pRasterizationState = &rasterizationStateCreateInfo;
    geometryPipelineCreateInfo.pMultisampleState = &multisampleStateCreateInfo;
    geometryPipelineCreateInfo.pDepthStencilState = &depthStencilStateCreateInfo;
    geometryPipelineCreateInfo.pColorBlendState = &colorBlendStateCreateInfo;
    geometryPipelineCreateInfo.pDynamicState = &dynamicStateCreateInfo;
    geometryPipelineCreateInfo.layout = _pipelineLayout;
    geometryPipelineCreateInfo.subpass = 0;
    geometryPipelineCreateInfo.basePipelineHandle = nullptr;
    geometryPipelineCreateInfo.basePipelineIndex = -1;

    vk::PipelineRenderingCreateInfoKHR pipelineRenderingCreateInfoKhr{};
    std::array<vk::Format, DEFERRED_ATTACHMENT_COUNT> formats{};
    std::fill(formats.begin(), formats.end(), GBuffers::GBufferFormat());
    pipelineRenderingCreateInfoKhr.colorAttachmentCount = DEFERRED_ATTACHMENT_COUNT;
    pipelineRenderingCreateInfoKhr.pColorAttachmentFormats = formats.data();
    pipelineRenderingCreateInfoKhr.depthAttachmentFormat = _gBuffers.DepthFormat();

    geometryPipelineCreateInfo.pNext = &pipelineRenderingCreateInfoKhr;
    geometryPipelineCreateInfo.renderPass = nullptr; // Using dynamic rendering.

    auto result = _brain.device.createGraphicsPipeline(nullptr, geometryPipelineCreateInfo, nullptr);
    util::VK_ASSERT(result.result, "Failed creating the geometry pipeline layout!");
    _pipeline = result.value;

    _brain.device.destroy(vertModule);
    _brain.device.destroy(fragModule);
}

void GeometryPipeline::CreateDescriptorSetLayout()
{
    std::array<vk::DescriptorSetLayoutBinding, 1> bindings{};

    vk::DescriptorSetLayoutBinding& descriptorSetLayoutBinding{bindings[0]};
    descriptorSetLayoutBinding.binding = 0;
    descriptorSetLayoutBinding.descriptorCount = 1;
    descriptorSetLayoutBinding.descriptorType = vk::DescriptorType::eUniformBufferDynamic;
    descriptorSetLayoutBinding.stageFlags = vk::ShaderStageFlagBits::eVertex;
    descriptorSetLayoutBinding.pImmutableSamplers = nullptr;

    vk::DescriptorSetLayoutCreateInfo createInfo{};
    createInfo.bindingCount = bindings.size();
    createInfo.pBindings = bindings.data();

    util::VK_ASSERT(_brain.device.createDescriptorSetLayout(&createInfo, nullptr, &_descriptorSetLayout),
                    "Failed creating geometry descriptor set layout!");
}

void GeometryPipeline::CreateDescriptorSets()
{
    std::array<vk::DescriptorSetLayout, MAX_FRAMES_IN_FLIGHT> layouts{};
    std::for_each(layouts.begin(), layouts.end(), [this](auto& l)
    { l = _descriptorSetLayout; });
    vk::DescriptorSetAllocateInfo allocateInfo{};
    allocateInfo.descriptorPool = _brain.descriptorPool;
    allocateInfo.descriptorSetCount = MAX_FRAMES_IN_FLIGHT;
    allocateInfo.pSetLayouts = layouts.data();

    std::array<vk::DescriptorSet, MAX_FRAMES_IN_FLIGHT> descriptorSets;

    util::VK_ASSERT(_brain.device.allocateDescriptorSets(&allocateInfo, descriptorSets.data()),
                    "Failed allocating descriptor sets!");
    for (size_t i = 0; i < descriptorSets.size(); ++i)
    {
        _frameData[i].descriptorSet = descriptorSets[i];
        UpdateGeometryDescriptorSet(i);
    }
}

void GeometryPipeline::UpdateGeometryDescriptorSet(uint32_t frameIndex)
{
    vk::DescriptorBufferInfo bufferInfo{};
    bufferInfo.buffer = _frameData[frameIndex].uniformBuffer;
    bufferInfo.offset = 0;
    bufferInfo.range = sizeof(UBO);

    std::array<vk::WriteDescriptorSet, 1> descriptorWrites{};

    vk::WriteDescriptorSet& bufferWrite{ descriptorWrites[0] };
    bufferWrite.dstSet = _frameData[frameIndex].descriptorSet;
    bufferWrite.dstBinding = 0;
    bufferWrite.dstArrayElement = 0;
    bufferWrite.descriptorType = vk::DescriptorType::eUniformBufferDynamic;
    bufferWrite.descriptorCount = 1;
    bufferWrite.pBufferInfo = &bufferInfo;

    _brain.device.updateDescriptorSets(descriptorWrites.size(), descriptorWrites.data(), 0, nullptr);
}

void GeometryPipeline::CreateUniformBuffers()
{
    vk::DeviceSize bufferSize = sizeof(UBO) * MAX_MESHES;

    for(size_t i = 0; i < _frameData.size(); ++i)
    {
        util::CreateBuffer(_brain, bufferSize,
                           vk::BufferUsageFlagBits::eUniformBuffer,
                           _frameData[i].uniformBuffer, true, _frameData[i].uniformBufferAllocation,
                           "Uniform buffer");

        util::VK_ASSERT(vmaMapMemory(_brain.vmaAllocator, _frameData[i].uniformBufferAllocation, &_frameData[i].uniformBufferMapped), "Failed mapping memory for UBO!");
    }
}

void GeometryPipeline::UpdateUniformData(uint32_t currentFrame, const std::vector<glm::mat4> transforms)
{
    static auto startTime = std::chrono::high_resolution_clock::now();

    auto currentTime = std::chrono::high_resolution_clock::now();
    float time = std::chrono::duration<float, std::chrono::seconds::period>(currentTime - startTime).count();

    glm::mat4 view = glm::lookAt(glm::vec3{sinf(time), 0.7f, cosf(time)}, glm::vec3{0.0f}, glm::vec3{0.0f, 1.0f, 0.0f});;
    glm::mat4 proj = glm::perspective(glm::radians(45.0f), _gBuffers.Size().x / static_cast<float>(_gBuffers.Size().y), 0.01f, 100.0f);
    proj[1][1] *= -1;

    std::array<UBO, MAX_MESHES> ubos;
    for(size_t i = 0; i < std::min(transforms.size(), ubos.size()); ++i)
    {
        ubos[i].model = transforms[i];
        ubos[i].view = view;
        ubos[i].proj = proj;
    }

    memcpy(_frameData[currentFrame].uniformBufferMapped, ubos.data(), ubos.size() * sizeof(UBO));
}
