#version 460

layout(set = 0, binding = 0) uniform UBO
{
    mat4 model;
} ubo;

layout(set = 1, binding = 0) uniform CameraUBO
{
    mat4 VP;
    mat4 view;
    mat4 proj;

    vec3 cameraPosition;
} cameraUbo;

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec4 inTangent;
layout(location = 3) in vec3 inColor;
layout(location = 4) in vec2 inTexCoord;

layout(location = 0) out vec3 position;
layout(location = 1) out vec3 normal;
layout(location = 2) out vec2 texCoord;

void main()
{
    position = (ubo.model * vec4(inPosition, 1.0)).xyz;
    normal = (ubo.model * vec4(inNormal, 0.0)).xyz;
    texCoord = inTexCoord;

    gl_Position = (cameraUbo.VP) * vec4(position, 1.0);
}