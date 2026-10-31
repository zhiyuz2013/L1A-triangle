#pragma once
#define GLM_FORCE_SWIZZLE
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <algorithm>
#include <cmath>
#include <vector>
#include <cstdint>
using namespace glm;
using uint = unsigned int;
#define layout(...)
#define uniform struct
#define buffer struct
#define readonly
#define in
#define out
#define xyz xyz()
#define rgb rgb()
#define discard return

inline int gl_VertexIndex = 0;
inline vec4 gl_Position{0.0f};
inline float gl_FragDepth = 1.0f;
inline uvec3 gl_GlobalInvocationID{0u};

