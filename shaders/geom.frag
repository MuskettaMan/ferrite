#version 460

layout(location = 0) in vec3 position;
layout(location = 1) in vec3 normal;
layout(location = 2) in vec2 texCoord;

layout(location = 0) out vec4 outAlbedoM;    // RGB: Albedo,   A: Metallic
layout(location = 1) out vec4 outNormalR;    // RGB: Normal,   A: Roughness
layout(location = 2) out vec4 outEmissiveAO; // RGB: Emissive, A: AO
layout(location = 3) out vec4 outPosition;   // RGB: Position, A: Unused

layout(set = 1, binding = 0) uniform sampler imageSampler;
layout(set = 1, binding = 1) uniform texture2D albedoImage;
layout(set = 1, binding = 2) uniform texture2D mrImage;
layout(set = 1, binding = 3) uniform texture2D normalImage;
layout(set = 1, binding = 4) uniform texture2D occlusionImage;
layout(set = 1, binding = 5) uniform texture2D emissiveImage;
layout(set = 1, binding = 6) uniform MaterialInfoUBO
{
    vec4 albedoFactor;

    float metallicFactor;
    float roughnessFactor;
    float normalScale;
    float occlusionStrength;

    bool useAlbedoMap;
    bool useMRMap;
    bool useNormalMap;
    bool useOcclusionMap;

    vec3 emissiveFactor;
    bool useEmissiveMap;
} materialInfoUBO;

void main()
{
    vec4 albedoSample = vec4(0.0);
    vec4 mrSample = vec4(0.0);
    vec4 occlusionSample = vec4(1.0);
    vec4 emissiveSample = vec4(0.0);
    if(materialInfoUBO.useAlbedoMap)
    {
        albedoSample = texture(sampler2D(albedoImage, imageSampler), texCoord);
    }
    if(materialInfoUBO.useMRMap)
    {
        mrSample = texture(sampler2D(mrImage, imageSampler), texCoord);
    }
    if(materialInfoUBO.useNormalMap)
    {
        // Can't sample normal map until we have TBN matrix.
    }
    if(materialInfoUBO.useOcclusionMap)
    {
        occlusionSample = texture(sampler2D(occlusionImage, imageSampler), texCoord);
    }
    if(materialInfoUBO.useEmissiveMap)
    {
        emissiveSample = texture(sampler2D(emissiveImage, imageSampler), texCoord);
    }

    outAlbedoM = vec4(albedoSample.rgb, mrSample.r);
    outEmissiveAO = vec4(emissiveSample.rgb, occlusionSample.r);
    outNormalR = vec4(normalize(normal), mrSample.g);

    outPosition = vec4(position, 1.0);
}