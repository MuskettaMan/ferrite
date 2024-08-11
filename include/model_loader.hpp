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

    vk::PrimitiveTopology MapGltfTopology(fastgltf::PrimitiveType gltfTopology);
    vk::IndexType MapIndexType(fastgltf::ComponentType componentType);
};
