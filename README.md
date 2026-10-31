# Vulkan Pyramid

A small native C++ sample that draws a colored 3D pyramid using the Vulkan API.
GLFW is used only to create the window and receive window events; rendering is
performed entirely with Vulkan (there is no OpenGL renderer).

## Requirements

- Linux with a working Vulkan driver and desktop session
- CMake 3.20 or newer
- A C++17 compiler
- Vulkan SDK headers/loader
- GLFW 3.3 or newer
- GLM
- `glslangValidator`

On Ubuntu, the development packages can be installed with:

```sh
sudo apt install build-essential cmake libvulkan-dev libglfw3-dev libglm-dev glslang-tools
```

Install a Vulkan driver for your GPU as well. For Mesa-based Intel/AMD GPUs,
`mesa-vulkan-drivers` provides one; NVIDIA systems need the matching proprietary
driver. `vulkan-tools` is optional but provides `vulkaninfo` for diagnostics.

## Build and run

From the project directory:

```sh
cmake -S . -B build
cmake --build build
./build/vulkan_triangle
```

In VS Code, use **Terminal > Run Build Task** (or `Ctrl+Shift+B`) to configure
and build the CMake target. To launch it from VS Code, run the
**Run Vulkan Triangle** task. The generated **C/C++: g++ build active file**
task is overridden in `.vscode/tasks.json` to run the same CMake build, since
compiling this source by itself does not compile its shaders or link Vulkan
and GLFW. To use **Run and Debug** or press `F5`, select **Debug Vulkan
Triangle**; its launch configuration builds the sample before starting it.

The program prints the Vulkan device it selected. It prefers a discrete GPU,
then an integrated GPU, and can use a CPU Vulkan implementation if that is the
only suitable device. Use **left-drag** to orbit, **middle/right-drag** to pan,
and **+/-** to zoom. **Ctrl+left-click** a visible point on the pyramid to set
the orbit and zoom pivot there; while holding **Ctrl**, left-drag to pan.
Press **R** to reset the camera, **P** for perspective projection, or **O**
for orthographic projection. Close the window to exit.
