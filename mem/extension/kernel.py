# MIT License

# Copyright (c) 2026 Mingjie Zhang (mingjie2026@gmail.com)

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Translate CUDA test includes in memory and compile them for ordinary GDB.

This is a serial CPU replay for independent CUDA threads, not a GPU simulator.
Unsupported synchronization must fail rather than silently change its meaning.
"""
from pathlib import Path
import argparse
import os
import re
import shlex
import shutil
import subprocess
import sys

from shader import extension_directories, line_directive, project_root


CUDA_HEADERS = {
    'cuda_runtime.h', 'cuda_runtime_api.h', 'device_launch_parameters.h',
    'device_atomic_functions.h',
}

# Reuse kernel.h's vector types and thread indices; provide host memory semantics.
CPU_RUNTIME = r"""
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#define __host__
#define __device__
#define __forceinline__ inline
using cudaError_t = int;
constexpr cudaError_t cudaSuccess = 0;
enum cudaMemcpyKind { cudaMemcpyHostToHost, cudaMemcpyHostToDevice,
                      cudaMemcpyDeviceToHost, cudaMemcpyDeviceToDevice };
template<class T> cudaError_t cudaMalloc(T** p, std::size_t n) {
    *p = static_cast<T*>(std::malloc(n)); return *p || !n ? cudaSuccess : 1;
}
inline cudaError_t cudaFree(void* p) { std::free(p); return cudaSuccess; }
inline cudaError_t cudaMemcpy(void* d, const void* s, std::size_t n, cudaMemcpyKind) {
    std::memcpy(d, s, n); return cudaSuccess;
}
inline cudaError_t cudaMemset(void* d, int value, std::size_t n) {
    std::memset(d, value, n); return cudaSuccess;
}
inline cudaError_t cudaDeviceSynchronize() { return cudaSuccess; }
inline cudaError_t cudaGetLastError() { return cudaSuccess; }
inline const char* cudaGetErrorString(cudaError_t e) { return e ? "CPU allocation failed" : "success"; }
template<class T> T atomicAdd(T* p, T value) { T old = *p; *p += value; return old; }
template<class T> T atomicSub(T* p, T value) { T old = *p; *p -= value; return old; }
inline void __syncthreads() { throw std::runtime_error("CPU replay does not support thread barriers"); }
inline thread_local dim3 gridDim;
template<class F> void mem_cpu_launch(dim3 grid, dim3 block, F function) {
    const dim3 savedBlock = blockDim, savedIndex = blockIdx, savedThread = threadIdx;
    const dim3 savedGrid = gridDim;
    gridDim = grid;
    blockDim = block;
    for (blockIdx.z = 0; blockIdx.z < grid.z; ++blockIdx.z)
    for (blockIdx.y = 0; blockIdx.y < grid.y; ++blockIdx.y)
    for (blockIdx.x = 0; blockIdx.x < grid.x; ++blockIdx.x)
    for (threadIdx.z = 0; threadIdx.z < block.z; ++threadIdx.z)
    for (threadIdx.y = 0; threadIdx.y < block.y; ++threadIdx.y)
    for (threadIdx.x = 0; threadIdx.x < block.x; ++threadIdx.x) function();
    gridDim = savedGrid;
    blockDim = savedBlock; blockIdx = savedIndex; threadIdx = savedThread;
}
"""


def masked(text):
    """Hide comments and literals while retaining offsets and line numbers."""
    return re.sub(r'/\*.*?\*/|//[^\n]*|"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'',
                  lambda m: re.sub(r'[^\n]', ' ' if m[0].startswith('/') else '@', m[0]),
                  text, flags=re.DOTALL)


def split_arguments(text):
    parts, start, depth = [], 0, 0
    for i, char in enumerate(masked(text)):
        if char in '([{':
            depth += 1
        elif char in ')]}':
            depth -= 1
        elif char == ',' and depth == 0:
            parts.append(text[start:i].strip())
            start = i + 1
    return parts + [text[start:].strip()]


def translate_kernel(text):
    """Convert launch syntax without changing ordinary C++ or comments."""
    clean = masked(text)
    if re.search(r'\b(?:__shared__|__syncthreads|__syncwarp)\b', clean):
        raise RuntimeError('Shared memory and thread barriers are not supported by serial CPU replay')
    pattern = re.compile(r'\b([A-Za-z_]\w*(?:::\w+)*)\s*<\s*<\s*<')
    replacements = []
    for match in pattern.finditer(clean):
        close = re.search(r'>\s*>\s*>\s*\(', clean[match.end():])
        if not close:
            raise RuntimeError('Unterminated CUDA launch configuration')
        config_end = match.end() + close.start()
        opening = match.end() + close.end() - 1
        end, depth = opening + 1, 1
        while end < len(clean) and depth:
            depth += (clean[end] == '(') - (clean[end] == ')')
            end += 1
        if depth:
            raise RuntimeError('Unterminated CUDA launch arguments')
        config = split_arguments(text[match.end():config_end])
        if len(config) < 2 or len(config) > 4:
            raise RuntimeError('Expected grid and block dimensions in CUDA launch')
        if any(arg not in ('0', 'nullptr') for arg in config[2:]):
            raise RuntimeError('CPU replay does not support dynamic shared memory or CUDA streams')
        replacement = (f'mem_cpu_launch(dim3({config[0]}), dim3({config[1]}), '
                       f'[&]() {{ {match[1]}({text[opening+1:end-1]}); }})')
        replacement += '\n' * max(0, text[match.start():end].count('\n') - replacement.count('\n'))
        replacements.append((match.start(), end, replacement))
    for start, end, replacement in reversed(replacements):
        text = text[:start] + replacement + text[end:]
    return text


class Translation:
    def __init__(self, source):
        self.source = source.resolve()
        self.root = project_root(self.source)
        self.directories = list(dict.fromkeys([
            self.source.parent, self.root, self.root / 'inc', self.root / 'include',
            *extension_directories(source)]))
        self.visited = set()
        self.pending = set()
        self.macros = set()

    def resolve(self, name, parent):
        return next((p.resolve() for p in [parent / name, *(d / name for d in self.directories)]
                     if p.is_file()), None)

    def companion(self, header):
        if header.suffix not in ('.h', '.hpp', '.hxx', '.cuh'):
            return
        for directory in (header.parent, self.root / 'src', self.root):
            for suffix in ('.cu', '.cpp', '.cc', '.cxx'):
                candidate = directory / (header.stem + suffix)
                if candidate.is_file() and candidate.resolve() != self.source:
                    self.pending.add(candidate.resolve())
                    return

    def expand(self, path):
        path = path.resolve()
        if path in self.visited:
            return ''
        self.visited.add(path)
        text = translate_kernel(path.read_text(encoding='utf-8'))
        # Only inspect real directives, never commented-out includes.
        clean = re.sub(r'/\*.*?\*/|//[^\n]*',
                       lambda m: re.sub(r'[^\n]', ' ', m[0]), text, flags=re.DOTALL)
        parts = [line_directive(1, path)]
        original_lines = text.splitlines(keepends=True)
        for number, (line, context) in enumerate(zip(original_lines, clean.splitlines(keepends=True)), 1):
            include = re.match(r'\s*#\s*include\s*[<"]([^>"\n]+)[>"]', context)
            if include:
                name = include[1]
                if name in CUDA_HEADERS or name == 'kernel.h':
                    parts.append('\n')
                    continue
                dependency = self.resolve(name, path.parent)
                if dependency:
                    self.companion(dependency)
                    parts.extend([self.expand(dependency), line_directive(number + 1, path)])
                    continue
            # Flattened source files may define the same helper macro differently.
            # Replace it explicitly instead of generating a redefinition warning.
            define = re.match(r'\s*#\s*define\s+(\w+)', context)
            if define:
                name = define[1]
                if name in self.macros:
                    parts.extend([f'#undef {name}\n', line_directive(number, path)])
                self.macros.add(name)
            if re.match(r'\s*#\s*pragma\s+once', context):
                parts.append('\n')
            else:
                parts.append(line)
        return ''.join(parts).rstrip('\n') + '\n'

    def program(self):
        header = next((d / 'kernel.h' for d in extension_directories(self.source)
                       if (d / 'kernel.h').is_file()), None)
        if header is None:
            raise FileNotFoundError('kernel.h not found')
        parts = [f'#include "{header.as_posix()}"\n', CPU_RUNTIME, self.expand(self.source)]
        while self.pending - self.visited:
            parts.append(self.expand(sorted(self.pending - self.visited)[0]))
        return ''.join(parts)


def build(source, executable):
    translation = Translation(source)
    program = translation.program()
    executable = executable.resolve()
    executable.parent.mkdir(parents=True, exist_ok=True)
    command = [*shlex.split(os.environ.get('CXX', 'g++')), '-std=c++17', '-O0', '-g3',
               '-x', 'c++', '-', '-o', str(executable)]
    return subprocess.run(command, input=program, encoding='utf-8').returncode


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--debugcpp', action='store_true')
    args = parser.parse_args()
    source = args.source.resolve()
    executable = (args.output or source.parent / 'build' / source.name.split('.')[0]).resolve()
    result = build(source, executable)
    if result or not args.debugcpp:
        return result
    debugger = shutil.which('gdb')
    if not debugger:
        raise RuntimeError('GDB is required for --debugcpp')
    return subprocess.run([debugger, '-q', '--args', str(executable)], cwd=source.parent).returncode


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (OSError, RuntimeError, ValueError) as error:
        print(f'kernel.py: {error}', file=sys.stderr)
        sys.exit(1)
