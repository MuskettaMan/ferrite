#version 460

layout(location = 0) in vec3 position;
layout(location = 1) in vec3 normal;
layout(location = 2) in vec2 texCoord;

layout(location = 0) out vec4 outAlbedo;
layout(location = 1) out vec4 outNormal;
layout(location = 2) out vec4 outPosition;

layout(binding = 1) uniform sampler imageSampler;
layout(binding = 2) uniform texture2D image;

void main()
{
    outAlbedo = texture(sampler2D(image, imageSampler), texCoord);
    outNormal = vec4(normalize(normal), 0.0);
    outPosition = vec4(position, 1.0);
}