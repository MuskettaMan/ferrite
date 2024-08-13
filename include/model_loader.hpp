#pragma once

#include "class_decorations.hpp"
#include "include.hpp"
#include "mesh.hpp"
#include <string>
#include <fastgltf/core.hpp>

class ModelLoader
{
public:
    ModelLoader(const VulkanBrain& brain, vk::DescriptorSetLayout materialDescriptorSetLayout);
    ~ModelLoader();

    NON_COPYABLE(ModelLoader);
    NON_MOVABLE(ModelLoader);

    ModelHandle Load(std::string_view path);

private:
    const VulkanBrain& _brain;
    fastgltf::Parser _parser;
    vk::Sampler _sampler;
    MaterialHandle _defaultMaterial;
    vk::DescriptorSetLayout _materialDescriptorSetLayout;

    Mesh ProcessMesh(const fastgltf::Mesh& gltfMesh, const fastgltf::Asset& gltf);
    MeshPrimitive ProcessPrimitive(const fastgltf::Primitive& primitive, const fastgltf::Asset& gltf);
    Texture ProcessImage(const fastgltf::Image& gltfImage, const fastgltf::Asset& gltf);
    Material ProcessMaterial(const fastgltf::Material& gltfMaterial, const fastgltf::Asset& gltf);

    vk::PrimitiveTopology MapGltfTopology(fastgltf::PrimitiveType gltfTopology);
    vk::IndexType MapIndexType(fastgltf::ComponentType componentType);
    uint32_t MapTextureIndexToImageIndex(uint32_t textureIndex, const fastgltf::Asset& gltf);

    void CalculateTangents(MeshPrimitive& primitive);
    glm::vec4 CalculateTangent(glm::vec3 p0, glm::vec3 p1, glm::vec3 p2, glm::vec2 uv0, glm::vec2 uv1, glm::vec2 uv2, glm::vec3 normal);
    ModelHandle LoadModel(const std::vector<Mesh>& meshes, const std::vector<Texture>& textures, const std::vector<Material>& materials, const fastgltf::Asset& gltf);

    void RecurseHierarchy(const fastgltf::Node& gltfNode, ModelHandle& hierarchy, const fastgltf::Asset& gltf, glm::mat4 matrix);
};
