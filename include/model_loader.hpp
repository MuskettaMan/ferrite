#pragma once

#include "class_decorations.hpp"
#include "mesh.hpp"
#include <string>
#include <fastgltf/core.hpp>

class ModelLoader
{
public:
    ModelLoader();

    NON_COPYABLE(ModelLoader);
    NON_MOVABLE(ModelLoader);

    Model Load(std::string_view path);

private:
    fastgltf::Parser _parser;

    Mesh ProcessMesh(const fastgltf::Mesh& gltfMesh, const fastgltf::Asset& gltf);
    MeshPrimitive ProcessPrimitive(const fastgltf::Primitive& primitive, const fastgltf::Asset& gltf);
    Texture ProcessImage(const fastgltf::Image& gltfImage, const fastgltf::Asset& gltf);

    vk::PrimitiveTopology MapGltfTopology(fastgltf::PrimitiveType gltfTopology);
    vk::IndexType MapIndexType(fastgltf::ComponentType componentType);

    void CalculateTangents(MeshPrimitive& primitive);
    glm::vec4 CalculateTangent(glm::vec3 p0, glm::vec3 p1, glm::vec3 p2, glm::vec2 uv0, glm::vec2 uv1, glm::vec2 uv2, glm::vec3 normal);
};
