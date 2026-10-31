#version 450

layout(push_constant) uniform Transform {
    mat4 mvp;
} transform;

layout(location = 0) out vec3 color;

const vec3 positions[18] = vec3[](
    vec3(0.0, 0.9, 0.0), vec3(-0.7, -0.7, -0.7), vec3(0.7, -0.7, -0.7),
    vec3(0.0, 0.9, 0.0), vec3(0.7, -0.7, -0.7), vec3(0.7, -0.7, 0.7),
    vec3(0.0, 0.9, 0.0), vec3(0.7, -0.7, 0.7), vec3(-0.7, -0.7, 0.7),
    vec3(0.0, 0.9, 0.0), vec3(-0.7, -0.7, 0.7), vec3(-0.7, -0.7, -0.7),
    vec3(-0.7, -0.7, -0.7), vec3(0.7, -0.7, 0.7), vec3(0.7, -0.7, -0.7),
    vec3(-0.7, -0.7, -0.7), vec3(-0.7, -0.7, 0.7), vec3(0.7, -0.7, 0.7)
);

const vec3 colors[18] = vec3[](
    vec3(1.0, 0.25, 0.18), vec3(1.0, 0.25, 0.18), vec3(1.0, 0.25, 0.18),
    vec3(0.2, 0.8, 0.95), vec3(0.2, 0.8, 0.95), vec3(0.2, 0.8, 0.95),
    vec3(0.95, 0.75, 0.2), vec3(0.95, 0.75, 0.2), vec3(0.95, 0.75, 0.2),
    vec3(0.55, 0.35, 0.95), vec3(0.55, 0.35, 0.95), vec3(0.55, 0.35, 0.95),
    vec3(0.2, 0.8, 0.4), vec3(0.2, 0.8, 0.4), vec3(0.2, 0.8, 0.4),
    vec3(0.2, 0.8, 0.4), vec3(0.2, 0.8, 0.4), vec3(0.2, 0.8, 0.4)
);

void main() {
    gl_Position = transform.mvp * vec4(positions[gl_VertexIndex], 1.0);
    color = colors[gl_VertexIndex];
}
