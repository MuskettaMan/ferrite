#version 460

layout(binding = 0) uniform sampler gBufferSampler;
layout(binding = 1) uniform texture2D gBufferAlbedoM;    // RGB: Albedo,   A: Metallic
layout(binding = 2) uniform texture2D gBufferNormalR;    // RGB: Normal,   A: Roughness
layout(binding = 3) uniform texture2D gBufferEmissiveAO; // RGB: Emissive, A: AO
layout(binding = 4) uniform texture2D gBufferPosition;   // RGB: Position, A: Unused

layout(location = 0) in vec2 texCoords;

layout(location = 0) out vec4 outColor;

void main()
{
    vec4 albedoM = texture(sampler2D(gBufferAlbedoM, gBufferSampler), texCoords);
    vec4 normalR = texture(sampler2D(gBufferNormalR, gBufferSampler), texCoords);
    vec4 emissiveAO = texture(sampler2D(gBufferEmissiveAO, gBufferSampler), texCoords);
    vec3 position = texture(sampler2D(gBufferPosition, gBufferSampler), texCoords).xyz;

    vec3 albedo = albedoM.rgb;
    float metallic = albedoM.a;
    vec3 normal = normalR.xyz;
    float roughness = normalR.a;
    vec3 emissive = emissiveAO.rgb;
    float ao = emissiveAO.a;

    if(normal == vec3(0.0, 0.0, 0.0))
        discard;

    vec3 lightDir = normalize(vec3(0.5));
    float diff = max(dot(normal, lightDir), 0.0);
    vec3 diffuse = diff * albedo;

    vec3 ambient = 0.1 * albedo;

    outColor = vec4(ambient + diffuse + emissive, 1.0) * ao;
}