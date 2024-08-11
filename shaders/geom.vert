#version 460

layout(binding = 0) uniform UBO
{
    mat4 model;
    mat4 view;
    mat4 proj;
} ubo;

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec3 inTangent;
layout(location = 3) in vec3 inColor;
layout(location = 4) in vec2 inTexCoord;

layout(location = 0) out vec3 position;
layout(location = 1) out vec3 normal;
layout(location = 2) out vec2 texCoord;

void main()
{
    position = vec3(ubo.model * vec4(inPosition, 1.0));
    normal = vec3(ubo.model * vec4(inPosition, 1.0));;
    texCoord = inTexCoord;

    gl_Position = (ubo.proj * ubo.view) * vec4(position, 1.0);
}