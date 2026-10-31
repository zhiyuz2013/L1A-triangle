
#pragma once

#include <cmath>
#include <cstddef>

// Host equivalents needed by the unchanged reconstruction kernel.
#define __constant__
#define __global__
struct float2 { float x, y; };
inline float2 make_float2(float x, float y) { return {x, y}; }
struct dim3 {
    unsigned int x, y, z;
    constexpr dim3(unsigned int x = 1, unsigned int y = 1, unsigned int z = 1)
        : x(x), y(y), z(z) {}
};
inline thread_local dim3 blockDim, blockIdx, threadIdx;
struct CpuTexture3D {
    const float* data;
    int width, height, depth;
};
using cudaTextureObject_t = const CpuTexture3D*;

// Unnormalized texel centers and zero border, with trilinear interpolation.
// CPU interpolation uses full floating point precision.
template <typename T>
inline T tex3D(cudaTextureObject_t texture, float x, float y, float z) {
    if (!texture || !std::isfinite(x) || !std::isfinite(y) || !std::isfinite(z)
        || x <= -0.5f || y <= -0.5f || z <= -0.5f
        || x >= texture->width + 0.5f || y >= texture->height + 0.5f
        || z >= texture->depth + 0.5f) return T{};
    x -= 0.5f; y -= 0.5f; z -= 0.5f;
    const int ix = static_cast<int>(std::floor(x));
    const int iy = static_cast<int>(std::floor(y));
    const int iz = static_cast<int>(std::floor(z));
    const float fx = x - ix, fy = y - iy, fz = z - iz;
    T result{};
    for (int dz = 0; dz < 2; ++dz)
        for (int dy = 0; dy < 2; ++dy)
            for (int dx = 0; dx < 2; ++dx) {
                const int cx = ix + dx, cy = iy + dy, cz = iz + dz;
                if (cx < 0 || cy < 0 || cz < 0 || cx >= texture->width
                    || cy >= texture->height || cz >= texture->depth) continue;
                const float weight = (dx ? fx : 1 - fx)
                    * (dy ? fy : 1 - fy) * (dz ? fz : 1 - fz);
                result += static_cast<T>(weight * texture->data[
                    (static_cast<std::size_t>(cz) * texture->height + cy)
                    * texture->width + cx]);
            }
    return result;
}

