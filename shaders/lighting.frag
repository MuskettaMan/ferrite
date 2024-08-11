#version 460
layout(binding = 0) uniform sampler gBufferSampler;
layout(binding = 1) uniform texture2D albedoBuffer;
layout(binding = 2) uniform texture2D normalBuffer;
layout(binding = 3) uniform texture2D positionBuffer;

layout(location = 0) in vec2 texCoords;

layout(location = 0) out vec4 outColor;

void main()
{
    vec3 position = texture(sampler2D(positionBuffer, gBufferSampler), texCoords).rgb;
    vec3 normal = texture(sampler2D(normalBuffer, gBufferSampler), texCoords).rgb;
    vec3 albedo = texture(sampler2D(albedoBuffer, gBufferSampler), texCoords).rgb;

    vec3 lightDir = normalize(vec3(0.5));
    float diff = max(dot(normal, lightDir), 0.0);
    vec3 diffuse = diff * albedo;

    vec3 ambient = 0.1 * albedo;

    outColor = vec4(ambient + diffuse, 1.0);
}