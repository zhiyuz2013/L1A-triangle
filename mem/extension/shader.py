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

from pathlib import Path
import os
import re
import shutil
import shlex
import subprocess
import sys

def extension_directories(source: Path):
    """Prefer the project's mem/extension, then the installed extension."""
    directories = []
    for parent in source.resolve().parents:
        local = parent / 'mem' / 'extension'
        if local.is_dir():
            directories.append(local)
            break
    directories.append(Path(__file__).resolve().parent)
    installed = os.environ.get('MEM_EXTENSION_PATH')
    if installed:
        directories.append(Path(installed).expanduser().resolve())
    # Standalone builds do not receive MEM_EXTENSION_PATH from VS Code.
    extensions = Path.home() / '.vscode' / 'extensions'
    directories.extend(sorted(extensions.glob('local.mem-*'),
                              key=lambda directory: directory.stat().st_mtime,
                              reverse=True))
    return list(dict.fromkeys(directories))


def project_root(source: Path):
    """Find project includes even when mem/extension has been removed."""
    source = source.resolve()
    configured = os.environ.get('PROJECT_ROOT')
    if configured:
        root = Path(configured).expanduser().resolve()
        if root in source.parents:
            return root
    return next((parent for parent in source.parents
                 if any((parent / marker).exists()
                        for marker in ('.vscode', '.git', 'CMakeLists.txt', 'mem/extension'))),
                source.parent)


def line_directive(number: int, path: Path):
    # line_directive(1, Path("shader/volume.frag")) -> #line 1 "shader/volume.frag"
    filename = str(path).replace('\\', '\\\\').replace('"', '\\"')
    return f'#line {number} "{filename}"\n'


def translate_storage_buffers(shader: str):
    """Give runtime-sized SSBO arrays owned storage for CPU replay.

    GLSL blocks without an instance expose their members globally. C++ macros
    cannot remove the block braces, so flatten those blocks here. Named blocks
    retain their structure and use shader.h's buffer/readonly/layout macros.
    Preserve newline counts for the original shader's GDB source locations.
    """
    # Mask comments for matching, but keep their text and source positions.
    masked = re.sub(r'/\*.*?\*/|//[^\n]*',
                    lambda m: re.sub(r'[^\n]', ' ', m.group()),
                    shader, flags=re.DOTALL)
    blocks = list(re.finditer(
        r'\bbuffer\s+\w+\s*\{(?P<body>[^{}]*)\}\s*(?P<instance>\w+)?\s*;',
        masked))
    for block in reversed(blocks):
        start, end = block.span('body')
        body = shader[start:end]
        # Only runtime arrays in storage blocks become vectors; other stage
        # interface arrays still use the existing pointer translation below.
        arrays = list(re.finditer(r'\b(\w+)\s+(\w+)\s*\[\s*\]\s*;',
                                  masked[start:end]))
        for array in reversed(arrays):
            declaration = f'std::vector<{array[1]}> {array[2]};'
            declaration += '\n' * array.group().count('\n')
            body = body[:array.start()] + declaration + body[array.end():]
        prefix = shader[block.start():start]
        suffix = shader[end:block.end()]
        if block['instance'] is None:
            prefix = re.sub(r'[^\n]', ' ', prefix)
            suffix = re.sub(r'[^\n]', ' ', suffix)
        shader = shader[:block.start()] + prefix + body + suffix + shader[block.end():]
    return shader


def translate_shader(shader_path: Path):
    shader = shader_path.read_text(encoding="utf-8")
    shader = translate_storage_buffers(shader)
    shader, entry_points = re.subn(r"\bvoid\s+(?:main|shader)\s*\(\s*\)", "void shader()", shader)
    if entry_points != 1:
        raise RuntimeError("Expected exactly one void main() or void shader() entry point in the shader")

    # GLSL decimal literals are floats; C++ otherwise promotes them to double.
    shader = re.sub(r'\b(\d+\.\d*(?:[eE][+-]?\d+)?)(?![\w.])', r'\1f', shader)
    # GLM function swizzles return values; update the original components directly.
    shader = re.sub(r'gl_Position\.xy\s*\+=\s*([^;]+);',
                    r'{ vec2 offset = \1; gl_Position.x += offset.x; gl_Position.y += offset.y; }', shader)
 
    shader = re.sub(r"^([ \t]*)#version\b", r"\1//version", shader, flags=re.MULTILINE)

    # const float arr[5] = float[](1.1, 1.2, 1.3, 1.4, 1.5); 
    # => const float arr[5] = {1.1, 1.2, 1.3, 1.4, 1.5};
    constructor = re.compile(r"\b(?:float|int|uint|bool|vec[234]|ivec[234]|uvec[234])\s*\[\s*\]\s*\(")
    while match := constructor.search(shader):
        end, depth = match.end(), 1
        while end < len(shader) and depth:
            depth += (shader[end] == '(') - (shader[end] == ')')
            end += 1
        if depth:
            raise RuntimeError("Unterminated GLSL array constructor")
        shader = shader[:match.start()] + '{' + shader[match.end():end - 1] + '}' + shader[end:]
    
    # vec4 position[]; => vec4* position = NULL;
    shader = re.sub(r"\b(\w+)\s+(\w+)\s*\[\s*\]\s*;", r"\1* \2 = NULL;", shader)
    return shader


def compile_in_memory(main_path: Path, shader_path: Path, command: list[str]):
    main_path = main_path.resolve()
    shader_path = shader_path.resolve()
    command = list(command)
    shader = translate_shader(shader_path)

    header = next((directory / 'shader.h' for directory in extension_directories(main_path)
                   if (directory / 'shader.h').is_file()), None)
    if header is None:
        raise FileNotFoundError('shader.h not found in mem/extension or the installed extension')
    parts = [line_directive(1, main_path)]
    replacements = 0
    for number, line in enumerate(main_path.read_text(encoding="utf-8").splitlines(keepends=True), 1):
        include = re.fullmatch(r'\s*#\s*include\s*"([^"\n]+)"\s*(?://[^\n]*)?\s*', line)
        if include and (main_path.parent / include.group(1)).resolve() == shader_path:
            parts.extend([f'#include "{header.as_posix()}"\n',
                          line_directive(1, shader_path), shader])
            if not shader.endswith("\n"):
                parts.append("\n")
            parts.append(line_directive(number + 1, main_path))
            replacements += 1
        elif include and include.group(1) == 'shader.h':
            # Older wrappers may still include this; use the selected header above.
            parts.append('\n')
        else:
            parts.append(line)
    if replacements != 1:
        raise RuntimeError(f'Expected exactly one shader include in {main_path}')

    # Keep CMake's flags, object output and dependency-generation arguments.
    source_index = next(
        index for index, arg in enumerate(command[1:], 1)
        if not arg.startswith("-") and Path(arg).resolve() == main_path
    )
    command[source_index:source_index + 1] = ["-x", "c++", "-", "-x", "none"]
    for directory in [*extension_directories(main_path), shader_path.parent, main_path.parent]:
        command.extend(["-iquote", str(directory)])
    return subprocess.run(command, input="".join(parts), encoding="utf-8").returncode

def find_shader(source: Path):
    """Resolve a shader from quoted includes relative to the C++ source."""
    shader_extensions = {'.vert', '.frag', '.geom', '.tesc', '.tese', '.comp', '.glsl'}
    shaders = []
    # Ignore commented-out includes, retaining newlines for line-based matching.
    context = re.sub(r'/\*.*?\*/|//[^\n]*',
                     lambda match: '\n' * match.group().count('\n'),
                     source.read_text(encoding='utf-8'), flags=re.DOTALL)
    for include in re.finditer(r'^\s*#\s*include\s*"([^"\n]+)"', context, re.MULTILINE):
        shader = (source.parent / include.group(1)).resolve()
        if shader.suffix.lower() in shader_extensions:
            if not shader.is_file():
                raise FileNotFoundError(f'Shader include does not exist: {shader}')
            shaders.append(shader)
    if len(shaders) > 1:
        raise RuntimeError(f'Expected at most one shader include in {source}')
    return shaders[0] if shaders else None


def referenced_sources(source: Path):
    """Follow local includes and compile their companion implementations only."""
    implementation_extensions = ('.cc', '.cpp', '.cxx', '.c')
    header_extensions = {'.h', '.hpp', '.hxx'}
    fallbacks = extension_directories(source)
    pending = [source.resolve()]
    visited, implementations, included = set(), set(), {source.resolve()}
    while pending:
        current = pending.pop()
        if current in visited:
            continue
        visited.add(current)
        text = re.sub(r'/\*.*?\*/|//[^\n]*',
                      lambda match: '\n' * match.group().count('\n'),
                      current.read_text(encoding='utf-8'), flags=re.DOTALL)
        for match in re.finditer(r'^\s*#\s*include\s*"([^"\n]+)"', text, re.MULTILINE):
            candidates = [*(directory / match[1] for directory in fallbacks),
                          current.parent / match[1], source.parent / match[1]]
            dependency = next((file.resolve() for file in candidates if file.is_file()), None)
            if dependency is None:
                continue  # Let the compiler report unresolved includes.
            pending.append(dependency)
            if dependency.suffix in implementation_extensions:
                included.add(dependency)
            if dependency.suffix in header_extensions:
                for extension in implementation_extensions:
                    companion = dependency.with_suffix(extension)
                    if companion.is_file():
                        implementations.add(companion)
                        pending.append(companion)
    return sorted(implementations - included)


def build(source: Path, executable: Path):
    source = source.resolve()
    executable.parent.mkdir(parents=True, exist_ok=True)
    shader = find_shader(source)
    command = [
            *shlex.split(os.environ.get("CXX", "g++")),
            "-std=c++17", "-O0", "-g3", str(source),
            *(str(file) for file in referenced_sources(source)),
            "-o", str(executable),
    ]
    if shader is not None:
        return compile_in_memory(source, shader, command)
    for directory in extension_directories(source):
        command.extend(['-iquote', str(directory)])
    return subprocess.run(command).returncode


def debugcpp(source: Path, executable: Path):
    """Build the supplied C++ program and debug it interactively in GDB."""
    debugger = shutil.which("gdb")
    if debugger is None:
        print("GDB is required to run debugcpp.", file=sys.stderr)
        return 1
    result = build(source, executable)
    if result:
        return result
    return subprocess.run([
        debugger, "-q",
        "-ex", "set debuginfod enabled off",
        "-ex", "break shader",
        "-ex", "run",
        "--args", str(executable),
    ], cwd=source.parent).returncode


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Build a C++ file using its shader includes.')
    parser.add_argument('source', type=Path)
    parser.add_argument('--output', type=Path)
    action = parser.add_mutually_exclusive_group()
    action.add_argument('--build', action='store_true', help='Build only (the default).')
    action.add_argument('--debugcpp', action='store_true', help='Build and launch GDB.')
    args = parser.parse_args()
    source = args.source.resolve()
    executable = (args.output or source.parent / 'build' / source.name.split('.')[0]).resolve()
    try:
        if args.debugcpp:
            return debugcpp(source, executable)
        return build(source, executable)
    except (OSError, RuntimeError, ValueError) as error:
        print(f'shader.py: {error}', file=sys.stderr)
        return 1

if __name__ == "__main__":
    sys.exit(main())
