#include "model_loader.hpp"
#include "spdlog/spdlog.h"
#include <fastgltf/tools.hpp>

ModelLoader::ModelLoader() : _parser()
{

}

Model ModelLoader::Load(std::string_view path)
{
    fastgltf::GltfFileStream fileStream{ path };

    if(!fileStream.isOpen())
        throw std::runtime_error("Path not found!");

    auto loadedGltf = _parser.loadGltf(fileStream, path, fastgltf::Options::DecomposeNodeMatrices);

    if(!loadedGltf)
        throw std::runtime_error(getErrorMessage(loadedGltf.error()).data());

    fastgltf::Asset& gltf = loadedGltf.get();

    if(gltf.scenes.size() > 1)
        spdlog::warn("GLTF contains more than one scene, but we only load one scene!");

    std::vector<Mesh> meshes{};

    for(auto& mesh : gltf.meshes)
    {
        meshes.emplace_back(ProcessMesh(mesh, gltf));
    }

    // TODO: Can be used for decoding the hierarchy.
//    fastgltf::iterateSceneNodes(gltf, 0, fastgltf::math::fmat4x4{}, [](fastgltf::Node& node, fastgltf::math::fmat4x4 nodeTransform) {
//
//    });

    return meshes;
}

Mesh ModelLoader::ProcessMesh(const fastgltf::Mesh& gltfMesh, const fastgltf::Asset& gltf)
{
    Mesh mesh{};

    for(auto& primitive : gltfMesh.primitives)
    {
        mesh.primitives.emplace_back(ProcessPrimitive(primitive, gltf));
    }

    return mesh;
}

MeshPrimitive ModelLoader::ProcessPrimitive(const fastgltf::Primitive& gltfPrimitive, const fastgltf::Asset& gltf)
{
    MeshPrimitive primitive{};

    primitive.topology = MapGltfTopology(gltfPrimitive.type);

    bool verticesReserved = false;

    for(auto& attribute : gltfPrimitive.attributes)
    {
        auto& accessor = gltf.accessors[attribute.accessorIndex];
        if(!accessor.bufferViewIndex.has_value())
            throw std::runtime_error("Failed retrieving buffer view index from accessor!");
        auto& bufferView = gltf.bufferViews[accessor.bufferViewIndex.value()];
        auto& buffer = gltf.buffers[bufferView.bufferIndex];
        auto& bufferBytes = std::get<fastgltf::sources::Array>(buffer.data);

        const std::byte* attributeBufferStart = bufferBytes.bytes.data() + bufferView.byteOffset + accessor.byteOffset;

        // Make sure the mesh primitive has enough space allocated.
        if(!verticesReserved)
        { primitive.vertices = std::vector<Vertex>(accessor.count); verticesReserved = true; }

        // TODO: Add support for normals.
        std::uint32_t offset;
        if(attribute.name == "POSITION")
            offset = offsetof(Vertex, position);
        else if(attribute.name == "TEXCOORD_0")
            offset = offsetof(Vertex, texCoord);
        else if(attribute.name == "COLOR0")
            offset = offsetof(Vertex, color);
        else
            continue;

        for(size_t i = 0; i < accessor.count; ++i)
        {
            const std::byte* element;
            if(bufferView.byteStride.has_value())
                element = attributeBufferStart + i * bufferView.byteStride.value();
            else
                element = attributeBufferStart + i * fastgltf::getElementByteSize(accessor.type, accessor.componentType);

            std::byte* writeTarget = reinterpret_cast<std::byte*>(&primitive.vertices[i]) + offset;
            std::memcpy(writeTarget, element, fastgltf::getElementByteSize(accessor.type, accessor.componentType));
        }
    }

    if(gltfPrimitive.indicesAccessor.has_value())
    {
        auto& accessor = gltf.accessors[gltfPrimitive.indicesAccessor.value()];
        if(!accessor.bufferViewIndex.has_value())
            throw std::runtime_error("Failed retrieving buffer view index from accessor!");
        auto& bufferView = gltf.bufferViews[accessor.bufferViewIndex.value()];
        auto& buffer = gltf.buffers[bufferView.bufferIndex];
        auto& bufferBytes = std::get<fastgltf::sources::Array>(buffer.data);

        uint32_t indexTypeSize = fastgltf::getElementByteSize(accessor.type, accessor.componentType);
        primitive.indexType = MapIndexType(accessor.componentType);
        primitive.indices = std::vector<std::byte>(accessor.count * indexTypeSize);

        const std::byte* attributeBufferStart = bufferBytes.bytes.data() + bufferView.byteOffset + accessor.byteOffset;

        if(!bufferView.byteStride.has_value() || bufferView.byteStride.value() == 0)
        {
            std::memcpy(primitive.indices.data(), attributeBufferStart, primitive.indices.size());
        }
        else
        {
            for(size_t i = 0; i < accessor.count; ++i)
            {
                const std::byte* element = attributeBufferStart + bufferView.byteStride.value() + i * indexTypeSize;
                std::byte* indexPtr = primitive.indices.data() + i * indexTypeSize;
                std::memcpy(indexPtr, element, indexTypeSize);
            }
        }
    }

    return primitive;
}

vk::PrimitiveTopology ModelLoader::MapGltfTopology(fastgltf::PrimitiveType gltfTopology)
{
    switch(gltfTopology)
    {
    case fastgltf::PrimitiveType::Points:        return vk::PrimitiveTopology::ePointList;
    case fastgltf::PrimitiveType::Lines:         return vk::PrimitiveTopology::eLineList;
    case fastgltf::PrimitiveType::LineLoop:      throw std::runtime_error("LineLoop isn't supported by Vulkan!");
    case fastgltf::PrimitiveType::LineStrip:     return vk::PrimitiveTopology::eLineStrip;
    case fastgltf::PrimitiveType::Triangles:     return vk::PrimitiveTopology::eTriangleList;
    case fastgltf::PrimitiveType::TriangleStrip: return vk::PrimitiveTopology::eTriangleStrip;
    case fastgltf::PrimitiveType::TriangleFan:   return vk::PrimitiveTopology::eTriangleFan;
    default: throw std::runtime_error("Unsupported primitive type!");
    }
}

vk::IndexType ModelLoader::MapIndexType(fastgltf::ComponentType componentType)
{
    switch(componentType)
    {
    case fastgltf::ComponentType::UnsignedInt:   return vk::IndexType::eUint32;
    case fastgltf::ComponentType::UnsignedShort: return vk::IndexType::eUint16;
    default: throw std::runtime_error("Unsupported index component type!");
    }
}

