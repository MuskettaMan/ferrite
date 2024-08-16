#include "model_loader.hpp"
#include "spdlog/spdlog.h"
#include <fastgltf/tools.hpp>
#include <fastgltf/glm_element_traits.hpp>
#include "stb_image.h"
#include "vulkan_helper.hpp"

ModelLoader::ModelLoader(const VulkanBrain& brain, vk::DescriptorSetLayout materialDescriptorSetLayout) :
    _brain(brain),
    _parser(),
    _materialDescriptorSetLayout(materialDescriptorSetLayout)
{

    _sampler = util::CreateSampler(_brain, vk::Filter::eLinear, vk::Filter::eLinear, vk::SamplerAddressMode::eRepeat, vk::SamplerMipmapMode::eLinear, static_cast<uint32_t>(floor(log2(2048))));

    Texture texture;
    texture.width = 2;
    texture.height = 2;
    texture.numChannels = 4;
    texture.data = std::vector<std::byte>(texture.width * texture.height * texture.numChannels * sizeof(float));
    std::array<std::shared_ptr<TextureHandle>, 5> textures;
    std::for_each(textures.begin(), textures.end(), [&texture, this](auto& ptr){
        ptr = std::make_shared<TextureHandle>();
        util::CreateTextureImage(_brain, texture, *ptr, false);
    });

    MaterialHandle::MaterialInfo info;
    _defaultMaterial = std::make_shared<MaterialHandle>(util::CreateMaterial(_brain, textures, info, *_sampler, _materialDescriptorSetLayout));
}

ModelLoader::~ModelLoader()
{
    vmaDestroyBuffer(_brain.vmaAllocator, _defaultMaterial->materialUniformBuffer, _defaultMaterial->materialUniformAllocation);
    for(auto& texture : _defaultMaterial->textures)
    {
        vmaDestroyImage(_brain.vmaAllocator, texture->image, texture->imageAllocation);
        _brain.device.destroy(texture->imageView);
    }
}

ModelHandle ModelLoader::Load(std::string_view path)
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

    std::vector<Mesh> meshes;
    std::vector<Texture> textures;
    std::vector<Material> materials;

    for(auto& mesh : gltf.meshes)
        meshes.emplace_back(ProcessMesh(mesh, gltf));

    for(auto& image : gltf.images)
        textures.emplace_back(ProcessImage(image, gltf));

    for(auto& material : gltf.materials)
        materials.emplace_back(ProcessMaterial(material, gltf));

    spdlog::info("Loaded model: {}", path);

    return LoadModel(meshes, textures, materials, gltf);
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
    if(gltfPrimitive.materialIndex.has_value())
        primitive.materialIndex = gltfPrimitive.materialIndex.value();

    bool verticesReserved = false;
    bool tangentFound = false;
    bool texCoordFound = false;

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

        std::uint32_t offset;
        if(attribute.name == "POSITION")
            offset = offsetof(Vertex, position);
        else if(attribute.name == "NORMAL")
            offset = offsetof(Vertex, normal);
        else if(attribute.name == "TANGENT")
        { offset = offsetof(Vertex, tangent); tangentFound = true; }
        else if(attribute.name == "TEXCOORD_0")
        { offset = offsetof(Vertex, texCoord); texCoordFound = true; }
        else if(attribute.name == "COLOR_0")
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
        primitive.indicesBytes = std::vector<std::byte>(accessor.count * indexTypeSize);

        const std::byte* attributeBufferStart = bufferBytes.bytes.data() + bufferView.byteOffset + accessor.byteOffset;

        if(!bufferView.byteStride.has_value() || bufferView.byteStride.value() == 0)
        {
            std::memcpy(primitive.indicesBytes.data(), attributeBufferStart, primitive.indicesBytes.size());
        }
        else
        {
            for(size_t i = 0; i < accessor.count; ++i)
            {
                const std::byte* element = attributeBufferStart + bufferView.byteStride.value() + i * indexTypeSize;
                std::byte* indexPtr = primitive.indicesBytes.data() + i * indexTypeSize;
                std::memcpy(indexPtr, element, indexTypeSize);
            }
        }
    }

    if(!tangentFound && texCoordFound)
        CalculateTangents(primitive);

    return primitive;
}

Texture ModelLoader::ProcessImage(const fastgltf::Image& gltfImage, const fastgltf::Asset& gltf)
{
    Texture texture{};

    std::visit(fastgltf::visitor {
            [](auto& arg) {},
            [&](const fastgltf::sources::URI& filePath) {
                assert(filePath.fileByteOffset == 0); // We don't support offsets with stbi.
                assert(filePath.uri.isLocalPath()); // We're only capable of loading local files.
                int32_t width, height, nrChannels;

                const std::string path(filePath.uri.path().begin(), filePath.uri.path().end()); // Thanks C++.
                stbi_uc* data = stbi_load(path.c_str(), &width, &height, &nrChannels, 4);
                if(!data) spdlog::error("Failed loading data from STBI at path: {}", path);

                texture.data = std::vector<std::byte>(width * height * 4);
                std::memcpy(texture.data.data(), reinterpret_cast<std::byte*>(data), texture.data.size());
                texture.width = width;
                texture.height = height;
                texture.numChannels = 4;

                stbi_image_free(data);
            },
            [&](const fastgltf::sources::Array& vector) {
                int32_t width, height, nrChannels;
                stbi_uc* data = stbi_load_from_memory(reinterpret_cast<const stbi_uc*>(vector.bytes.data()), static_cast<int32_t>(vector.bytes.size()), &width, &height, &nrChannels, 4);

                texture.data = std::vector<std::byte>(width * height * 4);
                std::memcpy(texture.data.data(), reinterpret_cast<std::byte*>(data), texture.data.size());
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

Material ModelLoader::ProcessMaterial(const fastgltf::Material& gltfMaterial, const fastgltf::Asset& gltf)
{
    Material material{};

    if(gltfMaterial.pbrData.baseColorTexture.has_value())
        material.albedoIndex = MapTextureIndexToImageIndex(gltfMaterial.pbrData.baseColorTexture.value().textureIndex, gltf);
    if(gltfMaterial.pbrData.metallicRoughnessTexture.has_value())
        material.metallicRoughnessIndex = MapTextureIndexToImageIndex(gltfMaterial.pbrData.metallicRoughnessTexture.value().textureIndex, gltf);
    if(gltfMaterial.normalTexture.has_value())
        material.normalIndex = MapTextureIndexToImageIndex(gltfMaterial.normalTexture.value().textureIndex, gltf);
    if(gltfMaterial.occlusionTexture.has_value())
        material.occlusionIndex = MapTextureIndexToImageIndex(gltfMaterial.occlusionTexture.value().textureIndex, gltf);
    if(gltfMaterial.emissiveTexture.has_value())
        material.emissiveIndex = MapTextureIndexToImageIndex(gltfMaterial.emissiveTexture.value().textureIndex, gltf);

    material.albedoFactor = *reinterpret_cast<const glm::vec4*>(&gltfMaterial.pbrData.baseColorFactor);
    material.metallicFactor = gltfMaterial.pbrData.metallicFactor;
    material.roughnessFactor = gltfMaterial.pbrData.roughnessFactor;
    material.normalScale = gltfMaterial.normalTexture.value().scale;
    material.emissiveFactor = *reinterpret_cast<const glm::vec3*>(&gltfMaterial.emissiveFactor);

    return material;
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

uint32_t ModelLoader::MapTextureIndexToImageIndex(uint32_t textureIndex, const fastgltf::Asset& gltf)
{
    return gltf.textures[textureIndex].imageIndex.value();
}

void ModelLoader::CalculateTangents(MeshPrimitive& primitive)
{
    uint32_t indexElementSize = (primitive.indexType == vk::IndexType::eUint16 ? 2 : 4);
    uint32_t triangleCount = primitive.indicesBytes.size() > 0 ? primitive.indicesBytes.size() / indexElementSize / 3 : primitive.vertices.size() / 3;
    for(size_t i = 0; i < triangleCount; ++i)
    {
        std::array<Vertex*, 3> triangle = {};
        if(primitive.indicesBytes.size() > 0)
        {
            std::array<uint32_t, 3> indices = {};
            std::memcpy(&indices[0], &primitive.indicesBytes[(i * 3 + 0) * indexElementSize], indexElementSize);
            std::memcpy(&indices[1], &primitive.indicesBytes[(i * 3 + 1) * indexElementSize], indexElementSize);
            std::memcpy(&indices[2], &primitive.indicesBytes[(i * 3 + 2) * indexElementSize], indexElementSize);

            triangle = {
                    &primitive.vertices[indices[0]],
                    &primitive.vertices[indices[1]],
                    &primitive.vertices[indices[2]]
            };
        }
        else
        {
            triangle = {
                    &primitive.vertices[i * 3 + 0],
                    &primitive.vertices[i * 3 + 1],
                    &primitive.vertices[i * 3 + 2]
            };
        }


        glm::vec4 tangent = CalculateTangent(triangle[0]->position, triangle[1]->position, triangle[2]->position,
                                             triangle[0]->texCoord, triangle[1]->texCoord, triangle[2]->texCoord,
                                             triangle[0]->normal);

        triangle[0]->tangent += tangent;
        triangle[1]->tangent += tangent;
        triangle[2]->tangent += tangent;
    }

    for(size_t i = 0; i < primitive.vertices.size(); ++i)
        primitive.vertices[i].tangent = glm::normalize(primitive.vertices[i].tangent);
}

glm::vec4 ModelLoader::CalculateTangent(glm::vec3 p0, glm::vec3 p1, glm::vec3 p2, glm::vec2 uv0, glm::vec2 uv1, glm::vec2 uv2, glm::vec3 normal)
{
    glm::vec3 e1 = p1 - p0;
    glm::vec3 e2 = p2 - p0;

    float deltaU1 = uv1.x - uv0.x;
    float deltaV1 = uv1.y - uv0.y;
    float deltaU2 = uv2.x - uv0.x;
    float deltaV2 = uv2.y - uv0.y;

    float f = 1.0f / (deltaU1 * deltaV2 - deltaU2 * deltaV1);

    glm::vec3 tangent;
    tangent = f * (deltaV2 * e1 - deltaV1 * e2);

    tangent = glm::normalize(tangent);

    glm::vec3 bitangent = glm::normalize(glm::cross(normal, tangent));
    float w = (glm::dot(glm::cross(normal, tangent), bitangent) < 0.0f) ? -1.0f : 1.0f;

    return glm::vec4(tangent.x, tangent.y, tangent.z, w);
}

ModelHandle ModelLoader::LoadModel(const std::vector<Mesh>& meshes, const std::vector<Texture>& textures, const std::vector<Material>& materials, const fastgltf::Asset& gltf)
{
    ModelHandle modelHandle{};

    // Load textures
    for(const auto& texture : textures)
    {
        TextureHandle textureHandle{};
        textureHandle.format = texture.GetFormat();
        textureHandle.width = texture.width;
        textureHandle.height = texture.height;

        util::CreateTextureImage(_brain, texture, textureHandle, true);

        modelHandle.textures.emplace_back(std::make_shared<TextureHandle>(textureHandle));
    }

    // Load materials
    for(const auto& material : materials)
    {
        std::array<std::shared_ptr<TextureHandle>, 5> textures;
        textures[0] = material.albedoIndex.has_value() ? modelHandle.textures[material.albedoIndex.value()] : nullptr;
        textures[1] = material.metallicRoughnessIndex.has_value() ? modelHandle.textures[material.metallicRoughnessIndex.value()] : nullptr;
        textures[2] = material.normalIndex.has_value() ? modelHandle.textures[material.normalIndex.value()] : nullptr;
        textures[3] = material.occlusionIndex.has_value() ? modelHandle.textures[material.occlusionIndex.value()] : nullptr;
        textures[4] = material.emissiveIndex.has_value() ? modelHandle.textures[material.emissiveIndex.value()] : nullptr;

        MaterialHandle::MaterialInfo info;
        info.useAlbedoMap = material.albedoIndex.has_value();
        info.useMRMap = material.metallicRoughnessIndex.has_value();
        info.useNormalMap = material.normalIndex.has_value();
        info.useOcclusionMap = material.occlusionIndex.has_value();
        info.useEmissiveMap = material.emissiveIndex.has_value();

        info.albedoFactor = material.albedoFactor;
        info.metallicFactor = material.metallicFactor;
        info.roughnessFactor = material.roughnessFactor;
        info.normalScale = material.normalScale;
        info.occlusionStrength = material.occlusionStrength;
        info.emissiveFactor = material.emissiveFactor;

        modelHandle.materials.emplace_back(std::make_shared<MaterialHandle>(util::CreateMaterial(_brain, textures, info, *_sampler, _materialDescriptorSetLayout, _defaultMaterial)));
    }

    // Load meshes
    for(const auto& mesh : meshes)
    {
        MeshHandle meshHandle{};

        for(const auto& primitive : mesh.primitives)
            meshHandle.primitives.emplace_back(LoadPrimitive(primitive, primitive.materialIndex.has_value() ? modelHandle.materials[primitive.materialIndex.value()] : nullptr));

        modelHandle.meshes.emplace_back(std::make_shared<MeshHandle>(meshHandle));
    }

    for(size_t i = 0; i < gltf.scenes[0].nodeIndices.size(); ++i)
        RecurseHierarchy(gltf.nodes[gltf.scenes[0].nodeIndices[i]], modelHandle, gltf, glm::mat4{1.0f});

    return modelHandle;
}

MeshPrimitiveHandle ModelLoader::LoadPrimitive(const MeshPrimitive& primitive, std::shared_ptr<MaterialHandle> material)
{
    MeshPrimitiveHandle primitiveHandle{};
    primitiveHandle.material = material == nullptr ? _defaultMaterial : material;
    primitiveHandle.topology = primitive.topology;
    primitiveHandle.indexType = primitive.indexType;
    primitiveHandle.indexCount = primitive.indicesBytes.size() / (primitiveHandle.indexType == vk::IndexType::eUint16 ? 2 : 4);

    util::CreateLocalBuffer(_brain, primitive.vertices, primitiveHandle.vertexBuffer, primitiveHandle.vertexBufferAllocation, vk::BufferUsageFlagBits::eVertexBuffer, "Vertex buffer");
    util::CreateLocalBuffer(_brain, primitive.indicesBytes, primitiveHandle.indexBuffer, primitiveHandle.indexBufferAllocation, vk::BufferUsageFlagBits::eIndexBuffer, "Index buffer");

    return primitiveHandle;
}

void ModelLoader::RecurseHierarchy(const fastgltf::Node& gltfNode, ModelHandle& modelHandle, const fastgltf::Asset& gltf, glm::mat4 matrix)
{
    Hierarchy::Node node{};

    if(gltfNode.meshIndex.has_value())
        node.mesh = modelHandle.meshes[gltfNode.meshIndex.value()];

    auto transform = fastgltf::getTransformMatrix(gltfNode, *reinterpret_cast<fastgltf::math::fmat4x4*>(&matrix));
    matrix = *reinterpret_cast<glm::mat4*>(&transform);
    node.transform = matrix;

    modelHandle.hierarchy.allNodes.emplace_back(node);

    for(size_t i = 0; i < gltfNode.children.size(); ++i)
    {
        RecurseHierarchy(gltf.nodes[gltfNode.children[i]], modelHandle, gltf, matrix);
    }
}

