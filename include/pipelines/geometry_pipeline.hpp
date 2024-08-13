#pragma once

#include "include.hpp"
#include "gbuffers.hpp"
#include "mesh.hpp"

struct UBO
{
    alignas(16) glm::mat4 model;
    alignas(16) glm::mat4 view;
    alignas(16) glm::mat4 proj;
};

constexpr uint32_t MAX_MESHES = 128;

class GeometryPipeline
{
public:
    GeometryPipeline(const VulkanBrain& brain, const GBuffers& gBuffers, vk::DescriptorSetLayout materialDescriptorSetLayout);
    ~GeometryPipeline();

    void RecordCommands(vk::CommandBuffer commandBuffer, uint32_t currentFrame, const ModelHandle& model);

    NON_MOVABLE(GeometryPipeline);
    NON_COPYABLE(GeometryPipeline);

private:
    struct FrameData
    {
        vk::Buffer uniformBuffer;
        VmaAllocation uniformBufferAllocation;
        void* uniformBufferMapped;
        vk::DescriptorSet descriptorSet;
    };

    void CreatePipeline(vk::DescriptorSetLayout materialDescriptorSetLayout);
    void CreateDescriptorSetLayout();
    void CreateDescriptorSets();
    void CreateUniformBuffers();
    void UpdateGeometryDescriptorSet(uint32_t frameIndex);
    void UpdateUniformData(uint32_t currentFrame, const std::vector<glm::mat4> transforms);

    const VulkanBrain& _brain;
    const GBuffers& _gBuffers;

    vk::DescriptorSetLayout _descriptorSetLayout;
    vk::PipelineLayout _pipelineLayout;
    vk::Pipeline _pipeline;

    std::array<FrameData, MAX_FRAMES_IN_FLIGHT> _frameData;
};
