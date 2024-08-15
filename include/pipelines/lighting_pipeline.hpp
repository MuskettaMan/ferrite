#pragma once

#include "include.hpp"
#include "gbuffers.hpp"
#include "mesh.hpp"
#include "swap_chain.hpp"
#include "hdr_target.hpp"

class LightingPipeline
{
public:
    LightingPipeline(const VulkanBrain& brain, const GBuffers& gBuffers, const HDRTarget& hdrTarget);
    ~LightingPipeline();

    void RecordCommands(vk::CommandBuffer commandBuffer, uint32_t currentFrame);
    void UpdateGBufferViews();

    NON_MOVABLE(LightingPipeline);
    NON_COPYABLE(LightingPipeline);

private:
    struct FrameData
    {
        vk::DescriptorSet descriptorSet;
    };

    void CreatePipeline();
    void CreateDescriptorSetLayout();
    void CreateDescriptorSets();

    const VulkanBrain& _brain;
    const GBuffers& _gBuffers;
    const HDRTarget& _hdrTarget;

    vk::DescriptorSetLayout _descriptorSetLayout;
    vk::PipelineLayout _pipelineLayout;
    vk::Pipeline _pipeline;

    vk::UniqueSampler _sampler;

    std::array<FrameData, MAX_FRAMES_IN_FLIGHT> _frameData;
};
