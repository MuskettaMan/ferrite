#include "mesh_primitives.hpp"
#include <concepts>

template<typename T>
concept indexType = std::same_as<T, uint16_t> || std::same_as<T, uint32_t>;

template <indexType T>
void AddTriangle(std::vector<std::byte>& indicesBytes, std::array<T, 3> triangle)
{
    size_t sizeBefore = indicesBytes.size();
    indicesBytes.resize(indicesBytes.size() + sizeof(triangle));
    std::memcpy(indicesBytes.data() + sizeBefore, triangle.data(), sizeof(triangle));
}

MeshPrimitive GenerateUVSphere(uint32_t slices, uint32_t stacks, float radius)
{
    MeshPrimitive primitive;

    uint32_t totalVertices = 2 + (stacks - 1) * slices;

    primitive.vertices.reserve(totalVertices);

    // TODO: Consider this based on the total amount of indices instead.
    primitive.indexType = vk::IndexType::eUint32;
    primitive.topology = vk::PrimitiveTopology::eTriangleList;
    primitive.materialIndex = std::nullopt;

    primitive.vertices.emplace_back(Vertex{ glm::vec3{ 0.0f, radius, 0.0f } });
    const uint32_t tipIndex = 0;

    for(uint32_t i = 0; i < stacks - 1; ++i)
    {
        float phi = glm::pi<float>() * static_cast<float>(i + 1) / stacks;
        for(uint32_t j = 0; j < slices; ++j)
        {
            float theta = 2.0f * glm::pi<float>() * static_cast<float>(j) / slices;
            glm::vec3 point{
                sinf(phi) * cosf(theta),
                cosf(phi),
                sinf(phi) * sinf(theta)
            };
            primitive.vertices.emplace_back(Vertex{ std::move(point * radius) });
        }
    }

    primitive.vertices.emplace_back(Vertex{ glm::vec3{ 0.0f, -radius, 0.0f } });
    const uint32_t bottomIndex = primitive.vertices.size() - 1;

    using Triangle = std::array<uint32_t, 3>;
    for(uint32_t i = 0; i < slices; ++i)
    {
        uint32_t i0 = i + 1;
        uint32_t i1 = (i + 1) % slices + 1;
        AddTriangle(primitive.indicesBytes, Triangle{ tipIndex, i1, i0 });

        i0 = i + slices * (stacks - 2) + 1;
        i1 = (i + 1) % slices + slices * (stacks - 2) + 1;
        AddTriangle(primitive.indicesBytes, Triangle{bottomIndex, i0, i1});
    }

    for(uint32_t j = 0; j < stacks - 2; j++)
    {
        uint32_t j0 = j * slices + 1;
        uint32_t j1 = (j + 1) * slices + 1;
        for(uint32_t i = 0; i < slices; i++)
        {
            uint32_t i0 = j0 + i;
            uint32_t i1 = j0 + (i + 1) % slices;
            uint32_t i2 = j1 + (i + 1) % slices;
            uint32_t i3 = j1 + i;

            AddTriangle(primitive.indicesBytes, Triangle{ i0, i1, i3 });
            AddTriangle(primitive.indicesBytes, Triangle{ i1, i2, i3 });
        }
    }

    return primitive;
}
