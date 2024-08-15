#version 460

layout(binding = 0) uniform sampler2D hdrTarget;

layout(location = 0) in vec2 texCoords;

layout(location = 0) out vec4 outColor;

vec3 aces(vec3 x) {
    const float a = 2.51;
    const float b = 0.03;
    const float c = 2.43;
    const float d = 0.59;
    const float e = 0.14;
    return clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0.0, 1.0);
}

void main()
{
    vec3 hdrColor = texture(hdrTarget, texCoords).rgb;

    // Reinhardt
    vec3 mapped = hdrColor / (hdrColor + vec3(1.0));

    // Aces
    //vec3 mapped = aces(hdrColor);

    outColor = vec4(mapped, 1.0);
}