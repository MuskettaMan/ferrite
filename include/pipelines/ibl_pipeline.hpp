#pragma once
#include "class_decorations.hpp"
#include "vulkan/vulkan.hpp"
#include "vk_mem_alloc.h"
#include "mesh.hpp"

struct VulkanBrain;
struct TextureHandle;

class IBLPipeline
{
public:
    IBLPipeline(const VulkanBrain& brain, const TextureHandle& environmentMap);
    ~IBLPipeline();

    void RecordCommands(vk::CommandBuffer commandBuffer);
    const Cubemap& IrradianceMap() const { return _irradianceMap; }

    NON_MOVABLE(IBLPipeline);
    NON_COPYABLE(IBLPipeline);

private:
    const VulkanBrain& _brain;
    const TextureHandle& _environmentMap;

    vk::PipelineLayout _pipelineLayout;
    vk::Pipeline _pipeline;
    vk::DescriptorSetLayout _descriptorSetLayout;
    vk::DescriptorSet _descriptorSet;

    Cubemap _irradianceMap;
    std::array<uint32_t, 6> _faceIndices = { 0, 1, 2, 3, 4, 5 };

    void CreatePipeline();
    void CreateDescriptorSetLayout();
    void CreateDescriptorSet();
};