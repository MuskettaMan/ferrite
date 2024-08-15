#pragma once

#include "include.hpp"
#include "gbuffers.hpp"
#include "mesh.hpp"
#include "swap_chain.hpp"

class LightingPipeline
{
public:
    LightingPipeline(const VulkanBrain& brain, const GBuffers& gBuffers, const SwapChain& swapChain);
    ~LightingPipeline();

    void RecordCommands(vk::CommandBuffer commandBuffer, uint32_t currentFrame, uint32_t swapChainIndex);
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
    const SwapChain& _swapChain;

    vk::DescriptorSetLayout _descriptorSetLayout;
    vk::PipelineLayout _pipelineLayout;
    vk::Pipeline _pipeline;

    vk::UniqueSampler _sampler;

    std::array<FrameData, MAX_FRAMES_IN_FLIGHT> _frameData;
};
