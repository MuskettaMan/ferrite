#pragma once

#include "include.hpp"
#include "swap_chain.hpp"
#include "vulkan_helper.hpp"

class SkydomePipeline
{
public:
    SkydomePipeline(const VulkanBrain& brain, const SwapChain& swapChain, MeshPrimitiveHandle&& sphere, const CameraStructure& camera);
    ~SkydomePipeline();

    void RecordCommands(vk::CommandBuffer commandBuffer, uint32_t currentFrame, uint32_t swapChainIndex);

    NON_COPYABLE(SkydomePipeline);
    NON_MOVABLE(SkydomePipeline);


private:
    const VulkanBrain& _brain;
    const SwapChain& _swapChain;
    const CameraStructure& _camera;

    MeshPrimitiveHandle _sphere;
    TextureHandle _hdri;
    vk::UniqueSampler _sampler;

    vk::PipelineLayout _pipelineLayout;
    vk::Pipeline _pipeline;
    vk::DescriptorSet _descriptorSet;
    vk::DescriptorSetLayout _descriptorSetLayout;

    void CreatePipeline();
    void CreateDescriptorSetLayout();
    void CreateDescriptorSet();
};