cmake_minimum_required(VERSION 3.20)
project(vulkan_triangle LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Default to a debuggable build while allowing -DCMAKE_BUILD_TYPE=Release.
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Debug CACHE STRING "Build type" FORCE)
endif()

find_package(Vulkan REQUIRED)
find_package(glfw3 3.3 REQUIRED)
find_package(glm CONFIG QUIET)
if(NOT TARGET glm::glm)
    find_path(GLM_INCLUDE_DIR glm/glm.hpp
        HINTS "${CMAKE_CURRENT_SOURCE_DIR}/../../../3rdParty/glm"
        REQUIRED
    )
    add_library(glm::glm INTERFACE IMPORTED)
    set_target_properties(glm::glm PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${GLM_INCLUDE_DIR}"
    )
endif()
find_program(GLSLANG_VALIDATOR glslangValidator REQUIRED)

add_executable(${PROJECT_NAME} src/main.cpp)
target_link_libraries(${PROJECT_NAME} PRIVATE Vulkan::Vulkan glfw glm::glm)
target_compile_definitions(${PROJECT_NAME} PRIVATE
    SHADER_DIR="${CMAKE_CURRENT_BINARY_DIR}/shaders"
    GLM_FORCE_DEPTH_ZERO_TO_ONE
)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(${PROJECT_NAME} PRIVATE
        -Wall -Wextra -Wpedantic -Wno-missing-field-initializers
        "$<$<CONFIG:Debug>:-O0;-g3>"
    )
endif()

# Compile GLSL to SPIR-V in the build directory with debug information.
set(SHADER_OUTPUT_DIR "${CMAKE_CURRENT_BINARY_DIR}/shaders")
file(MAKE_DIRECTORY "${SHADER_OUTPUT_DIR}")

foreach(SHADER triangle.vert triangle.frag)
    add_custom_command(
        OUTPUT "${SHADER_OUTPUT_DIR}/${SHADER}.spv"
        COMMAND "${GLSLANG_VALIDATOR}" -V --glsl-version 450 -g -Od
                "${CMAKE_CURRENT_SOURCE_DIR}/shaders/${SHADER}"
                -o "${SHADER_OUTPUT_DIR}/${SHADER}.spv"
        DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/shaders/${SHADER}"
        VERBATIM
    )
    list(APPEND SHADER_BINARIES "${SHADER_OUTPUT_DIR}/${SHADER}.spv")
endforeach()

add_custom_target(vulkan_shaders DEPENDS ${SHADER_BINARIES})
add_dependencies(${PROJECT_NAME} vulkan_shaders)
