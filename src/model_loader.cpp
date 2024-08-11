#include "model_loader.hpp"
#include "spdlog/spdlog.h"
#include <fastgltf/tools.hpp>
#include "stb_image.h"

ModelLoader::ModelLoader() : _parser()
{

}

Model ModelLoader::Load(std::string_view path)
{
    fastgltf::GltfFileStream fileStream{ path };

    if(!fileStream.isOpen())
        throw std::runtime_error("Path not found!");

    std::string_view directory = path.substr(0, path.find_last_of('/'));
    auto loadedGltf = _parser.loadGltf(fileStream, directory, fastgltf::Options::DecomposeNodeMatrices | fastgltf::Options::LoadExternalBuffers | fastgltf::Options::LoadExternalImages);

    if(!loadedGltf)
        throw std::runtime_error(getErrorMessage(loadedGltf.error()).data());

    fastgltf::Asset& gltf = loadedGltf.get();

    if(gltf.scenes.size() > 1)
        spdlog::warn("GLTF contains more than one scene, but we only load one scene!");

    Model model{};

    for(auto& mesh : gltf.meshes)
        model.meshes.emplace_back(ProcessMesh(mesh, gltf));

    for(auto& image : gltf.images)
        model.textures.emplace_back(ProcessImage(image, gltf));

    // TODO: Can be used for decoding the hierarchy.
//    fastgltf::iterateSceneNodes(gltf, 0, fastgltf::math::fmat4x4{}, [](fastgltf::Node& node, fastgltf::math::fmat4x4 nodeTransform) {
//
//    });

    return model;
}

Mesh ModelLoader::ProcessMesh(const fastgltf::Mesh& gltfMesh, const fastgltf::Asset& gltf)
{
    Mesh mesh{};

    for(auto& primitive : gltfMesh.primitives)
        mesh.primitives.emplace_back(ProcessPrimitive(primitive, gltf));

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

Texture ModelLoader::ProcessImage(const fastgltf::Image& gltfImage, const fastgltf::Asset& gltf)
{
    Texture texture{};

    auto handleStbiData = [&](stbi_uc* data, uint32_t dataSize)
    {
        texture.data = std::vector<std::byte>(dataSize);
        std::copy(texture.data.begin(), texture.data.end(), reinterpret_cast<std::byte*>(data));
    };

    std::visit(fastgltf::visitor {
            [](auto& arg) {},
            [&](const fastgltf::sources::URI& filePath) {
                assert(filePath.fileByteOffset == 0); // We don't support offsets with stbi.
                assert(filePath.uri.isLocalPath()); // We're only capable of loading local files.
                int32_t width, height, nrChannels;

                const std::string path(filePath.uri.path().begin(), filePath.uri.path().end()); // Thanks C++.
                stbi_uc* data = stbi_load(path.c_str(), &width, &height, &nrChannels, 4);
                if(!data) spdlog::error("Failed loading data from STBI at path: {}", path);

                handleStbiData(data, width * height * 4);
                texture.width = width;
                texture.height = height;
                texture.numChannels = 4;

                stbi_image_free(data);
            },
            [&](const fastgltf::sources::Array& vector) {
                int32_t width, height, nrChannels;
                stbi_uc* data = stbi_load_from_memory(reinterpret_cast<const stbi_uc*>(vector.bytes.data()), static_cast<int32_t>(vector.bytes.size()), &width, &height, &nrChannels, 4);

                handleStbiData(data, width * height * 4);
                texture.width = width;
                texture.height = height;
                texture.numChannels = 4;

                stbi_image_free(data);
            },
            [&](const fastgltf::sources::BufferView& view) {
                auto& bufferView = gltf.bufferViews[view.bufferViewIndex];
                auto& buffer = gltf.buffers[bufferView.bufferIndex];

                std::visit(fastgltf::visitor {
                        // We only care about VectorWithMime here, because we specify LoadExternalBuffers, meaning
                        // all buffers are already loaded into a vector.
                        [](auto& arg) {},
                        [&](const fastgltf::sources::Array& vector) {
                            int32_t width, height, nrChannels;
                            stbi_uc* data = stbi_load_from_memory(reinterpret_cast<const stbi_uc*>(vector.bytes.data() + bufferView.byteOffset),
                                                                        static_cast<int32_t>(bufferView.byteLength), &width, &height, &nrChannels, 4);

                            texture.data = std::vector<std::byte>(width * height * 4);
                            std::memcpy(texture.data.data(), reinterpret_cast<std::byte*>(data), texture.data.size());
                            texture.width = width;
                            texture.height = height;
                            texture.numChannels = 4;

                            stbi_image_free(data);
                        }
                }, buffer.data);
            },
    }, gltfImage.data);

    return texture;
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

