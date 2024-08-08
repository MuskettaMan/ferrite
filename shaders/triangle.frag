#version 460

layout(location = 0) in vec3 fragColor;
layout(location = 1) in vec2 texCoord;

layout(location = 0) out vec4 outColor;

layout(binding = 1) uniform sampler2D imageSampler;

void main()
{
    vec3 color = fragColor;
    color = texture(imageSampler, texCoord).rgb;

    outColor = vec4(pow(color, vec3(1.0 / 2.2)), 1.0);
}