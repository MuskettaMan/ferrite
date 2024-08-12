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

void main()
{
    vec4 mr = texture(sampler2D(mrImage, imageSampler), texCoord);

    outAlbedoM.rgb = texture(sampler2D(albedoImage, imageSampler), texCoord).rgb;
    outAlbedoM.a = mr.g;

    outNormalR.rgb = normalize(normal);
    outNormalR.a = mr.b;

    outEmissiveAO.rgb = texture(sampler2D(emissiveImage, imageSampler), texCoord).rgb;
    outEmissiveAO.a = texture(sampler2D(occlusionImage, imageSampler), texCoord).r;

    outPosition = vec4(position, 1.0);
}