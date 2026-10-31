// MIT License

// Copyright (c) 2026 Mingjie Zhang (mingjie2026@gmail.com)

// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

LOG = 0; // 10/29 2026

const vscode = require('vscode');
const { registerShaderDefinitions } = require('./definitions');
const { vscodeSettings } = require('./settings');
const path = require('path');
const fs = require('fs');
const { execFile } = require('child_process');
const { promisify } = require('util');
const execFileAsync = promisify(execFile);

function log(...args) {
    const stack = new Error().stack;
    const m = stack.split('\n')[2]?.match(/:(\d+):\d+\)?$/);
    const line = m ? m[1] : '';

    vscode.debug.activeDebugConsole.appendLine(
        `>${line} ${args.join(' ')}`
    );
}

async function printStack(session) {
    const editor = vscode.window.activeTextEditor;
    const line = editor.selection.active.line + 1;

    let stack;
    try {
        LOG && log('printStack', line);
        const item = vscode.debug.activeStackItem;
        let threadId = item && item.session === session ? item.threadId : undefined;
        // CUDA-GDB owns a separate device focus. Asking for the first host
        // thread's stack can replace that focus before evaluating GPU locals.
        if (session.type === 'cuda-gdb') return;
        if (threadId === undefined) {
            const response = await session.customRequest('threads');
            threadId = response.threads && response.threads[0] && response.threads[0].id;
        }
        if (threadId === undefined) {
            log("printStack, threadId === undefined, Error");
        } else {
            const response = await session.customRequest('stackTrace', {
                threadId,
                startFrame: 0,
                levels: 4
            });
            stack = (response.stackFrames || []).slice(0, 4)
                .map(frame => frame.name.replace(/\(.*$/, '').trim())
                .join(' <-- ');
        }
        vscode.debug.activeDebugConsole.appendLine(`${line} ${stack}`);        
    } catch (e) {
        log('Error printStack', line, stack, e.message);
    }
}

async function getFrameId(session) {
    // Let Nsight evaluate commands in its current CUDA focus. A DAP frame
    // selection can switch to a host architecture, which cannot read CUDA
    // DWARF registers (including the 0x40000000 register range).
    if (session.type === 'cuda-gdb') return undefined;
    const item = vscode.debug.activeStackItem;
    if (item && item.session === session && typeof item.frameId === 'number') {
        return item.frameId;
    }
    try {
        let tid = item && item.session === session && typeof item.threadId === 'number' ? item.threadId : undefined;
        if (tid === undefined) {
            const threads = await session.customRequest('threads');
            tid = threads && threads.threads && threads.threads[0] && threads.threads[0].id;
        }
        if (tid === undefined) return undefined;
        const st = await session.customRequest('stackTrace', {
            threadId: tid, startFrame: 0, levels: 1
        });
        return st && st.stackFrames && st.stackFrames[0] && st.stackFrames[0].id;
    } catch (e) {
        return undefined;
    }
}

async function openingProject(context) {
    let openingProject = false;
    const openProject = async (editor) => {
        if (openingProject) return false;
        openingProject = true;
        try {
            return await openProjectFromEditor(editor);
        } catch (e) {
            log('Error opening project folder', e.message);
            return false;
        } finally {
            openingProject = false;
        }
    };
    context.subscriptions.push(
        vscode.window.onDidChangeActiveTextEditor(openProject)
    );
    return await openProject(vscode.window.activeTextEditor);
}

async function initialMem(session) {
    const editor = vscode.window.activeTextEditor;
    const frameId = await getFrameId(session);
    LOG && log('initialMem', 'getFrameId', frameId);

    const workspaceFolders = vscode.workspace.workspaceFolders || [];
    const folder = editor ? vscode.workspace.getWorkspaceFolder(editor.document.uri) : undefined;

    const rootFolder = [session.workspaceFolder, folder, ...workspaceFolders]
        .filter(Boolean)
        .find(workspaceFolder => {
            const settingsPath = path.join(workspaceFolder.uri.fsPath, '.vscode');
            return fs.existsSync(settingsPath) && fs.statSync(settingsPath).isDirectory();
        });

    const PROJECT_ROOT = (rootFolder || session.workspaceFolder || folder)?.uri.fsPath;
    if (!PROJECT_ROOT) {
        throw new Error('Cannot determine PROJECT_ROOT: open a project workspace folder.');
    }

    const project_root = JSON.stringify(PROJECT_ROOT.replace(/\\/g, '/'));
    const extension_path = JSON.stringify(__dirname.replace(/\\/g, '/'));

    const expr = `-exec python import os, sys; PROJECT_ROOT=${project_root}; os.environ['PROJECT_ROOT']=PROJECT_ROOT; sys.path[:0]=[os.path.join(PROJECT_ROOT, 'mem/extension'), ${extension_path}]; import importlib, mem; importlib.reload(mem); from mem import *;`; // NOMODIFY, NODELETE
    LOG && log('initialMem', expr);

    try {
        const response = await session.customRequest('evaluate', {
            expression: expr,
            context: 'repl',
            frameId: frameId
        });
        if (response && response.result) {
            vscode.debug.activeDebugConsole.appendLine(response.result);
        }
    } catch (e) {
        log('Error initialMem', expr, e.message);
        throw e;
    }
}

async function runPython(editor) {
    try {
        const folder = vscode.workspace.getWorkspaceFolder(editor.document.uri);
        const started = await vscode.debug.startDebugging(folder, {
            name: 'Python: Current File (Ctrl+F8)',
            type: 'debugpy',
            request: 'launch',
            program: editor.document.uri.fsPath,
            cwd: path.dirname(editor.document.uri.fsPath),
            console: 'internalConsole',
            internalConsoleOptions: 'openOnSessionStart',
            redirectOutput: true,
            noDebug: true
        });
        if (!started) {
            vscode.window.showErrorMessage('mem: Could not run the Python file.');
        }
    } catch (e) {
        log('Error runPython', e.message);
        vscode.window.showErrorMessage(`mem: ${e.message}`);
    }
}

function tokenText() {
    const editor = vscode.window.activeTextEditor;
    const pos = editor.selection.active;

    const range = editor.document.getWordRangeAtPosition(pos);
    LOG && log('memCppCpp', 'pos', pos, 'range', range);
    if (!range)
        return ""

    let text = editor.document.getText(range);
    let left = editor.document.lineAt(range.start.line).text.slice(0, range.start.character);
    let match;
    while ((match = left.match(/([A-Za-z_][A-Za-z0-9_]*)\s*(\.|->)\s*$/))) {
        text = match[1] + match[2] + text;
        left = left.slice(0, match.index);
    }

    LOG && log('tokenText', pos.line + 1, text);
    return text;
}

async function requestExpr(expr) {
    const session = vscode.debug.activeDebugSession;
    const frameId = await getFrameId(session);

    try {
        const response = await session.customRequest('evaluate', {
            expression: expr,
            context: 'repl',
            frameId: frameId
        });
        if (response && response.result) {
            vscode.debug.activeDebugConsole.appendLine(response.result);
        }
    } catch (e) {
        log('Error requestExpr', expr, e.message);
    }
}

async function runExec_source(session, sourceText) {
    const sourceArgs = sourceText.trim().match(/^(?:"([^"]+)"|'([^']+)'|(\S+))(?:\s+([A-Za-z_]\w*)(?:\s+(.+))?)?$/);
    if (!sourceArgs) {
        vscode.window.showErrorMessage('mem: Use source <file> [function_name [arguments...]]. Quote paths containing spaces.');
        return true;
    }
    const filename = sourceArgs[1] || sourceArgs[2] || sourceArgs[3];
    const functionName = sourceArgs[4];
    const functionArgs = sourceArgs[5] ? sourceArgs[5].trim().split(/\s+/) : [];
    if (functionName && !/\.py$/i.test(filename)) {
        vscode.window.showErrorMessage('mem: Function calls require a Python (.py) script.');
        return true;
    }
    const editor = vscode.window.activeTextEditor;
    const currentPath = editor ? path.dirname(editor.document.uri.fsPath) : undefined;
    const folder = editor && vscode.workspace.getWorkspaceFolder(editor.document.uri);
    const projectPath = (session.workspaceFolder || folder)?.uri.fsPath
        || process.env.PROJECT_ROOT || currentPath;
    const relativePath = filename.replace(/^[/\\]+/, '');
    const candidates = [
        currentPath && path.resolve(currentPath, filename),
        path.isAbsolute(filename) && filename,
        projectPath && path.join(projectPath, relativePath),
        projectPath && path.join(projectPath, 'src', relativePath)
    ].filter(Boolean);
    const sourcePath = candidates.find(candidate =>
        fs.existsSync(candidate) && fs.statSync(candidate).isFile());
    if (!sourcePath) {
        vscode.window.showErrorMessage(`mem: Script not found: ${filename}`);
        return true;
    }
    const normalizedPath = sourcePath.replace(/\\/g, '/');
    const expr = functionName
        ? `-exec python __file__ = ${JSON.stringify(normalizedPath)}; exec(compile(open(__file__, 'rb').read(), __file__, 'exec'), globals()); ${functionName}(*${JSON.stringify(functionArgs)})`
        : `-exec source ${normalizedPath}`;
    const frameId = await getFrameId(session);
    LOG && log(expr);
    const response = await session.customRequest('evaluate', {
        expression: expr,
        context: 'repl',
        frameId: frameId
    });
    if (response && response.result) {
        vscode.debug.activeDebugConsole.appendLine(response.result);
    }
    return true;
}

async function runExec(text) {
    const session = vscode.debug.activeDebugSession;
    if (!session) {
        vscode.window.showErrorMessage('mem: Start a debug session before running -exec commands.');
        return true;
    }

    text = text.trim().replace(/^(?:\/{2,}|\/\*)\s*/, '')
        .replace(/\s*\*\/\s*$/, '').trim();
    // Ignore standalone source annotations when executing debugger commands.
    text = text.replace(/\b(?:READONLY|NODELETE|NOMODIFY)\b/g, '')
        .replace(/\s*(?:\/\/|#)[\s,]*$/, '').trim();
    const match = text.match(/^-?exec\s+(.+)$/);
    if (!match) return false;

    let command = match[1];
    const source = command.match(/^source\s+(.+)$/);
    if (source) {
        return await runExec_source(session, source[1]);
    }
    else if (command.match(/^break\b/)) {
        const editor = vscode.window.activeTextEditor;
        if (editor && !editor.document.isClosed && !await editor.document.save()) return true;
        try {
            LOG && log("toggleBreakpoint");
            await toggleBreakpoint();
        } catch (error) {
            vscode.debug.activeDebugConsole.appendLine(`toggleBreakpoint: ${error.message || error}`);
        }
    }
    else{
        const expr = `-exec ${command}`;
        const frameId = await getFrameId(session);
        LOG && log(expr);
        const response = await session.customRequest('evaluate', {
            expression: expr,
            context: 'repl',
            frameId: frameId
        });
        if (response && response.result) {
            vscode.debug.activeDebugConsole.appendLine(response.result);
        }
    }
    return true;
}

async function aware(session) {
    const frameId = await getFrameId(session);

    const kernelDimensions = [
        "import gdb, re",
        "import numpy as np",
        "try:",
        "    kernels_output = gdb.execute('info cuda kernels', to_string=True)",
        "",
        "    kernels_match = re.search(",
        "        r'^\\s*\\*[^\\n]*?(\\(\\d+\\s*,\\s*\\d+\\s*,\\s*\\d+\\))\\s+'",
        "        r'(\\(\\d+\\s*,\\s*\\d+\\s*,\\s*\\d+\\))',",
        "        kernels_output, re.MULTILINE)",
        "",
        "    BLOCK, KERNEL = (",
        "        [np.array([int(value) for value in group.strip('()').split(',')])",
        "         for group in kernels_match.groups()]",
        "        if kernels_match else (np.zeros(3, dtype=int), np.zeros(3, dtype=int))",
        "    )",
        "",
        "    kernel_output = gdb.execute('cuda block thread', to_string=True)",
        "",
        "    kernel_match = re.search(",
        "        r'(\\(\\d+\\s*,\\s*\\d+\\s*,\\s*\\d+\\)),?\\s+(?:thread\\s+)?'",
        "        r'(\\(\\d+\\s*,\\s*\\d+\\s*,\\s*\\d+\\))',",
        "        kernel_output, re.MULTILINE)",
        "",
        "    block, kernel = (",
        "        [np.array([int(value) for value in group.strip('()').split(',')])",
        "         for group in kernel_match.groups()]",
        "        if kernel_match else (np.zeros(3, dtype=int), np.zeros(3, dtype=int))",
        "    )",
        "",
        "    id = block * KERNEL + kernel if kernels_match and kernel_match else None",
        "",
        "    with np.printoptions(formatter={'int_kind': lambda value: str(value)}):",
        "        print(id, '/', block, kernel, '/', BLOCK, KERNEL, '<-- kernel')",
        "except gdb.error:",
        "    pass"
    ].join('\n');
    try {
        const expr = `-exec python exec(${JSON.stringify(kernelDimensions)})`;
        const response = await session.customRequest('evaluate', {
            expression: expr,
            context: 'repl',
            frameId: frameId
        });
        if (response && typeof response.result === 'string' && response.result.length > 0) {
            vscode.debug.activeDebugConsole.appendLine(response.result);
        }
    } catch (error) {
        vscode.debug.activeDebugConsole.appendLine(`aware(): ${error.message || error}`);
    }
    return true;
}

async function memCppCpp(session) {
    const editor = vscode.window.activeTextEditor;
    const line = editor.selection.active.line + 1;
    const frameId = await getFrameId(session);
    let text;
    if (editor.selection.isEmpty) { 
        text = editor.document.lineAt(editor.selection.active.line).text;
        if(!await runExec(text)){  // -EXEC NODELETE         
            text = tokenText();
            if (text){ // TOKEN NODELETE
                const expression = `value = mem(${JSON.stringify(text)}); printf(value, ${JSON.stringify('<= ' + text)}, list(value.shape) if hasattr(value, 'shape') and value.shape else '', '|${line}|')`;
                const expr = `-exec python exec(${JSON.stringify(expression)})`.replace(/ {2,}/g, ' ');
                const response = await session.customRequest('evaluate', {
                    expression: expr,
                    context: 'repl',
                    frameId: frameId
                });
                if (response && typeof response.result === 'string' && response.result.length > 0) {
                    vscode.debug.activeDebugConsole.appendLine(response.result);
                }
            }
        }
    } else { // SELECTION
        text = editor.document.getText(editor.selection);
        const isParameter = /^\S+$/.test(text.trim());
        
        if(isParameter){
            const expression = `value = mem(${JSON.stringify(text)})\nprintf(value, ${JSON.stringify('<= ' + text)}, list(value.shape) if hasattr(value, 'shape') and value.shape else '', '|${line}|')`;
            const expr = `-exec python exec(${JSON.stringify(expression)})`;
            const response = await session.customRequest('evaluate', {
                expression: expr,
                context: 'repl',
                frameId: frameId
            });
            if (response && typeof response.result === 'string' && response.result.length > 0) {
                vscode.debug.activeDebugConsole.appendLine(response.result);
            }
        }
        else{
            const lines = text.split(/\r?\n/).map(line => line.replace(/\s+$/, ''));

            const valid_lines = [];
            for (const line of lines.map(line => line.trim())){
                if(line !== '' && !await runExec(line)){ // -EXEC NODELETE  
                    valid_lines.push(line);
                }
            }
            
            if(valid_lines.length > 0){
                let expression = valid_lines.join(" \n");
                const expr = `-exec python exec(${JSON.stringify(expression)})`;
                const response = await session.customRequest('evaluate', {
                    expression: expr,
                    context: 'repl',
                    frameId: frameId
                });
                if (response && typeof response.result === 'string' && response.result.length > 0) {
                    vscode.debug.activeDebugConsole.appendLine(response.result);
                }
            }
        }
    }
}

async function memCppPython(session) {
    const editor = vscode.window.activeTextEditor;
    const line = editor.selection.active.line + 1;
    const frameId = await getFrameId(session);

    let text;
    if (editor.selection.isEmpty) { // ALL TEXT NODELETE
        text = editor.document.getText(); 
        LOG && log('memCppPython', 'editor.document.getText()', text);
    } else { // SELECTION NODELETE
        text = editor.document.getText(editor.selection);
        LOG && log('memCppPython', 'editor.document.getText(editor.selection)', text);
    }

    const lines = text.split(/\r?\n/).map(line => line.replace(/\s+$/, ''));
    expression = lines.join(' \n');
    expression = JSON.stringify(expression);

    const expr = `-exec python exec(${expression})`;
    LOG && log('memCppPython', expr);

    try {
        const response = await session.customRequest('evaluate', {
            expression: expr,
            context: 'repl',
            frameId: frameId
        });
        // if (response && typeof response.result === 'string' && response.result.length > 0) {
        //     vscode.debug.activeDebugConsole.appendLine(response.result);
        // }
    } catch (e) {
        log('Error memCppPython', expr, e.message);
    }
}

async function memPythonPython(session) {
    const editor = vscode.window.activeTextEditor;
    // if (!editor.selection.isEmpty) {
    //     await vscode.commands.executeCommand('editor.debug.action.selectionToRepl');
    //     return;
    // }
    
    const frameId = await getFrameId(session);
    let expression;

    if (editor.selection.isEmpty) {
        const pos = editor.selection.active;
        const range = editor.document.getWordRangeAtPosition(pos);
        if (range) { // TOKEN
            const text = editor.document.getText(range);
            LOG && log('memCppPython TOKEN', 'range', JSON.stringify(range));
            LOG && log('memCppPython TOKEN', 'text', text);

            const inspectCode = [
                'import numpy as np',
                'display = value',
                "size = ''",
                'try:',
                '    data = np.asarray(value)',
                '    if isinstance(value, (list, tuple)) or hasattr(value, "shape"):',
                "        size = list(data.shape) if data.shape else ''",
                '    if np.issubdtype(data.dtype, np.number):',
                '        display = data.round(3) if data.shape else data.round(3).item()',
                '    if isinstance(display, np.ndarray):',
                '        display = np.array2string(display, max_line_width=200, threshold=100, edgeitems=50, precision=3, suppress_small=True)',
                'except (TypeError, ValueError):',
                '    size = "ragged" if isinstance(value, (list, tuple)) else ""',
                'lines = str(display).splitlines()',
                'if len(lines) > 20:',
                '    display = "\\n".join(lines[:10] + ["..."] + lines[-10:])',
                'print(display, label, size, location)'
            ].join('\n');
            expression = `exec(${JSON.stringify(inspectCode)}, {'value': ${text}, 'label': ${JSON.stringify(`<= ${text}`)}, 'location': ${JSON.stringify(`⌊${pos.line + 1}⌉`)}})`;
     
        } else { // ALL TEXT
            expression = editor.document.getText(); 
            LOG && log('memPythonPython', 'editor.document.getText()', expression);

        }
    } else { // SELECTION
        expression = editor.document.getText(editor.selection);
        LOG && log('memPythonPython', 'editor.document.getText(editor.selection)', expression);
    }

    const expr = expression;
    LOG && log('memPythonPython', expr);

    try {
        const response = await session.customRequest('evaluate', {
            expression: expr,
            context: 'repl',
            frameId: frameId
        });
        if (response && typeof response.result === 'string' && response.result.length > 0) {
            vscode.debug.activeDebugConsole.appendLine(response.result);
        }
    } catch (e) {
        log('Error memPythonPython', expr, e.message);
    }
}

async function openProjectFromEditor(editor) {
    if (vscode.workspace.workspaceFolders?.length || !editor
        || editor.document.uri.scheme !== 'file') return false;

    let directory = path.dirname(editor.document.uri.fsPath);
    while (true) {
        try {
            if (fs.statSync(path.join(directory, '.vscode')).isDirectory()) {
                await vscode.commands.executeCommand(
                    'vscode.openFolder', vscode.Uri.file(directory), false
                );
                return true;
            }
        } catch (e) {
            if (e.code !== 'ENOENT' && e.code !== 'ENOTDIR') {
                log('Error finding project folder', e.message);
                return false;
            }
        }
        const parent = path.dirname(directory);
        if (parent === directory) return false;
        directory = parent;
    }
}

async function debugPython(editor) {
    try {
        const folder = vscode.workspace.getWorkspaceFolder(editor.document.uri);
        const started = await vscode.debug.startDebugging(folder, {
            name: 'Python: Current File (F8)',
            type: 'debugpy',
            request: 'launch',
            program: editor.document.uri.fsPath,
            cwd: path.dirname(editor.document.uri.fsPath),
            console: 'internalConsole',
            internalConsoleOptions: 'openOnSessionStart',
            redirectOutput: true,
            noDebug: false
        });
        if (!started) {
            vscode.window.showErrorMessage('mem: Could not start the Python debugger.');
        }
    } catch (e) {
        log('Error debugPython', e.message);
        vscode.window.showErrorMessage(`mem: ${e.message}`);
    }
}

function appBuildDirectory(editor) {
    const folder = vscode.workspace.getWorkspaceFolder(editor.document.uri);
    const root = folder ? folder.uri.fsPath : path.dirname(editor.document.uri.fsPath);
    return path.join(root, 'build');
}

function appExecutable(editor) {
    const programName = path.basename(editor.document.uri.fsPath).split('.')[0];
    if (!programName) throw new Error('The filename must contain a program name before the first dot.');
    return path.join(appBuildDirectory(editor), programName);
}

let buildErrorTerminal;
let buildErrorWriter;

async function showBuildError(error) {
    const output = [error.message, error.stdout, error.stderr]
        .filter(Boolean).map(value => String(value).trimEnd()).join('\n');
    if (vscode.debug.activeDebugSession) {
        vscode.debug.activeDebugConsole.appendLine(output);
        await vscode.commands.executeCommand('workbench.panel.repl.view.focus');
    } else {
        // F6 can fail before a debug session exists. A terminal keeps that
        // compiler output visible even when there is no active Debug Console.
        if (!buildErrorTerminal) {
            buildErrorWriter = new vscode.EventEmitter();
            buildErrorTerminal = vscode.window.createTerminal({
                name: 'mem: Build errors',
                pty: {
                    onDidWrite: buildErrorWriter.event,
                    open() {
                        buildErrorWriter.fire(output.replace(/\r?\n/g, '\r\n') + '\r\n');
                    },
                    close() {
                        buildErrorTerminal = undefined;
                        buildErrorWriter.dispose();
                        buildErrorWriter = undefined;
                    }
                }
            });
        } else {
            buildErrorWriter.fire(output.replace(/\r?\n/g, '\r\n') + '\r\n');
        }
        buildErrorTerminal.show();
    }
    vscode.window.showErrorMessage('mem: Build failed. See the compiler output.');
}

async function buildCpp(editor, source = editor.document.uri.fsPath) {
    try {
        if (!await vscode.workspace.saveAll()) return false;
        const filename = editor.document.uri.fsPath;
        const folder = vscode.workspace.getWorkspaceFolder(editor.document.uri);
        const root = folder ? folder.uri.fsPath : path.dirname(filename);
        await fs.promises.mkdir(appBuildDirectory(editor), { recursive: true });
        const { stdout, stderr } = await execFileAsync(
            process.env.CXX || 'g++',
            ['-std=c++17', '-O0', '-g3', source, '-o', appExecutable(editor)],
            { cwd: root, maxBuffer: 10 * 1024 * 1024 }
        );
        if (stdout) vscode.debug.activeDebugConsole.appendLine(stdout.trimEnd());
        if (stderr) vscode.debug.activeDebugConsole.appendLine(stderr.trimEnd());
        const started = await vscode.debug.startDebugging(folder, {
            name: `C++: Debug ${path.basename(filename)}`,
            type: 'cppdbg',
            request: 'launch',
            program: appExecutable(editor),
            cwd: path.dirname(source),
            MIMode: 'gdb',
            miDebuggerPath: process.env.GDB || 'gdb',
            externalConsole: false,
            internalConsoleOptions: 'openOnSessionStart',
            stopAtEntry: false
        });
        if (!started) {
            vscode.window.showErrorMessage('mem: Could not start the C++ debugger.');
        }
        // Debugging is handled here; skip the caller's launch.json configuration.
        return false;
    } catch (e) {
        await showBuildError(e);
        return false;
    }
}

async function buildShader(editor, source) {
    try {
        if (!await vscode.workspace.saveAll()) return false;
        const filename = editor.document.uri.fsPath;
        source = source || (/\.(cpp|cc|cxx|c|h|hpp|hxx)$/i.test(filename)
            ? filename : `${filename}_.cc`);
        const workspace = vscode.workspace.getWorkspaceFolder(editor.document.uri);
        const root = workspace ? workspace.uri.fsPath : path.dirname(source);
        let builder = path.join(__dirname, 'shader.py');
        const localBuilder = path.join(root, 'mem', 'extension', 'shader.py');
        if (fs.existsSync(localBuilder)) builder = localBuilder;
        await fs.promises.mkdir(appBuildDirectory(editor), { recursive: true });
        const { stdout, stderr } = await execFileAsync(
            'python3',
            [builder, source, '--output', appExecutable(editor)],
            { cwd: path.dirname(source), maxBuffer: 10 * 1024 * 1024,
                env: { ...process.env, MEM_EXTENSION_PATH: __dirname, PROJECT_ROOT: root } }
        );
        if (stdout) vscode.debug.activeDebugConsole.appendLine(stdout.trimEnd());
        if (stderr) vscode.debug.activeDebugConsole.appendLine(stderr.trimEnd());
        const folder = vscode.workspace.getWorkspaceFolder(editor.document.uri);
        const started = await vscode.debug.startDebugging(folder, {
            name: `Shader: Debug ${path.basename(filename)}`,
            type: 'cppdbg',
            request: 'launch',
            program: appExecutable(editor),
            cwd: path.dirname(source),
            MIMode: 'gdb',
            miDebuggerPath: process.env.GDB || 'gdb',
            externalConsole: false,
            internalConsoleOptions: 'openOnSessionStart',
            stopAtEntry: false
        });
        if (!started) {
            vscode.window.showErrorMessage('mem: Could not start the shader debugger.');
        }
        // Debugging is handled here; stop the caller from invoking debugCpp,
        // which reads launch.json and can run the application's build task.
        return false;
    } catch (e) {
        await showBuildError(e);
        return false;
    }
}

async function buildKernel(editor, source) {
    try {
        if (!await vscode.workspace.saveAll()) return false;
        const filename = editor.document.uri.fsPath;
        source = source || (/\.(cpp|cc|cxx|c|h|hpp|hxx)$/i.test(filename)
            ? filename : `${filename}_.cc`);
        source = path.resolve(source);
        const workspace = vscode.workspace.getWorkspaceFolder(editor.document.uri);
        const root = workspace ? workspace.uri.fsPath : path.dirname(source);
        let builder = path.join(__dirname, 'kernel.py');
        const localBuilder = path.join(root, 'mem', 'extension', 'kernel.py');
        if (fs.existsSync(localBuilder)) builder = localBuilder;
        await fs.promises.mkdir(appBuildDirectory(editor), { recursive: true });
        const { stdout, stderr } = await execFileAsync(
            'python3',
            [builder, source, '--output', appExecutable(editor)],
            { cwd: path.dirname(source), maxBuffer: 10 * 1024 * 1024,
                env: { ...process.env, MEM_EXTENSION_PATH: __dirname, PROJECT_ROOT: root } }
        );
        if (stdout) vscode.debug.activeDebugConsole.appendLine(stdout.trimEnd());
        if (stderr) vscode.debug.activeDebugConsole.appendLine(stderr.trimEnd());
        const folder = vscode.workspace.getWorkspaceFolder(editor.document.uri);
        const started = await vscode.debug.startDebugging(folder, {
            name: `Kernel: Debug ${path.basename(filename)}`,
            type: 'cppdbg',
            request: 'launch',
            program: appExecutable(editor),
            cwd: path.dirname(source),
            MIMode: 'gdb',
            miDebuggerPath: process.env.GDB || 'gdb',
            externalConsole: false,
            internalConsoleOptions: 'openOnSessionStart',
            stopAtEntry: false
        });
        if (!started) {
            vscode.window.showErrorMessage('mem: Could not start the kernel debugger.');
        }
        // Debugging is handled here; stop the caller from invoking debugCpp,
        // which reads launch.json and can run the application's build task.
        return false;
    } catch (e) {
        await showBuildError(e);
        return false;
    }
}

// Blank the original entry point while preserving source line numbers.
function omitKernelMain(source) {
    const clean = source.replace(
        /R"([^\s()\\]{0,16})\([\s\S]*?\)\1"|\/\*[\s\S]*?\*\/|\/\/[^\r\n]*|"(?:\\[\s\S]|[^"\\])*"|'(?:\\[\s\S]|[^'\\])*'|^[ \t]*#.*(?:\\\r?\n.*)*$/gm,
        text => text.replace(/[^\r\n]/g, ' ')
    );
    const ranges = [];
    for (const match of clean.matchAll(/\b(?:int|auto)\s+main\s*\(/g)) {
        let end = match.index + match[0].length;
        let depth = 1;
        while (end < clean.length && depth) {
            if (clean[end] === '(') depth++;
            if (clean[end] === ')') depth--;
            end++;
        }
        const body = /^\s*(?:noexcept\s*(?:\([^)]*\))?\s*)?(?:->\s*int\s*)?\{/.exec(clean.slice(end));
        if (depth || !body) continue;
        end += body[0].length;
        depth = 1;
        while (end < clean.length && depth) {
            if (clean[end] === '{') depth++;
            if (clean[end] === '}') depth--;
            end++;
        }
        if (depth) throw new Error('mem: Unclosed main function.');
        ranges.push([match.index, end]);
    }
    for (const [start, end] of ranges.reverse()) {
        source = source.slice(0, start)
            + source.slice(start, end).replace(/[^\r\n]/g, ' ')
            + source.slice(end);
    }
    return source;
}

async function buildKernelMain(editor, source) {
    try {
        if (!await vscode.workspace.saveAll()) return false;
        const filename = editor.document.uri.fsPath;
        source = source || (/\.(cpp|cc|cxx|c|h|hpp|hxx)$/i.test(filename)
            ? filename : `${filename}_.cc`);
        const workspace = vscode.workspace.getWorkspaceFolder(editor.document.uri);
        const root = workspace ? workspace.uri.fsPath : path.dirname(source);
        const localHeaders = path.join(root, 'mem', 'extension');
        const headerDirectory = fs.existsSync(path.join(localHeaders, 'kernel.h'))
            ? localHeaders : __dirname;
        await fs.promises.mkdir(appBuildDirectory(editor), { recursive: true });
        const sourceText = omitKernelMain(await fs.promises.readFile(source, 'utf8'));
        const input = `#line 1 ${JSON.stringify(path.resolve(source))}\n${sourceText}\n`
            + '#line 1 "<mem-kernel-entry>"\nint main() { return 0; }\n';
        const compilation = execFileAsync(
            process.env.CXX || 'g++',
            ['-std=c++17', '-O0', '-g3', '-x', 'c++',
                '-I', headerDirectory, '-include', path.join(headerDirectory, 'kernel.h'),
                '-I', path.dirname(path.resolve(source)), '-', '-o', appExecutable(editor)],
            { cwd: path.dirname(source), maxBuffer: 10 * 1024 * 1024,
                env: { ...process.env, MEM_EXTENSION_PATH: __dirname, PROJECT_ROOT: root } }
        );
        // Feed only the transformed copy to the compiler; never rewrite source.
        compilation.child.stdin.on('error', () => {}); // Compiler failures are reported below.
        compilation.child.stdin.end(input);
        const { stdout, stderr } = await compilation;
        if (stdout) vscode.debug.activeDebugConsole.appendLine(stdout.trimEnd());
        if (stderr) vscode.debug.activeDebugConsole.appendLine(stderr.trimEnd());
        const folder = vscode.workspace.getWorkspaceFolder(editor.document.uri);
        const started = await vscode.debug.startDebugging(folder, {
            name: `Kernel: Debug ${path.basename(filename)}`,
            type: 'cppdbg',
            request: 'launch',
            program: appExecutable(editor),
            cwd: path.dirname(source),
            MIMode: 'gdb',
            miDebuggerPath: process.env.GDB || 'gdb',
            externalConsole: false,
            internalConsoleOptions: 'openOnSessionStart',
            stopAtEntry: false
        });
        if (!started) {
            vscode.window.showErrorMessage('mem: Could not start the kernel debugger.');
        }
        // Debugging is handled here; stop the caller from invoking debugCpp,
        // which reads launch.json and can run the application's build task.
        return false;
    } catch (e) {
        await showBuildError(e);
        return false;
    }
}

async function debugCpp(editor, debuggerType = 'cppdbg') {
    try {
        const programName = path.basename(editor.document.uri.fsPath).split('.')[0];
        if (!programName) {
            vscode.window.showErrorMessage('mem: The filename must contain a program name before the first dot.');
            return;
        }
        const folder = vscode.workspace.getWorkspaceFolder(editor.document.uri);
        if (!folder) {
            vscode.window.showErrorMessage('mem: Open the project folder before debugging C++.');
            return;
        }
        const configurations = vscode.workspace
            .getConfiguration('launch', folder.uri)
            .get('configurations', []);
        const configuration = configurations.find(config =>
            config.type === debuggerType && config.request === 'launch'
        );
        const defaultConfiguration = {
            type: debuggerType,
            request: 'launch',
            cwd: folder.uri.fsPath,
            ...(debuggerType === 'cppdbg' ? {
                MIMode: 'gdb',
                miDebuggerPath: process.env.GDB || 'gdb',
                externalConsole: false
            } : {
                debuggerPath: process.env.CUDA_GDB || '/usr/local/cuda/bin/cuda-gdb'
            })
        };
        if (!await vscode.workspace.saveAll()) return;
        const started = await vscode.debug.startDebugging(folder, {
            ...defaultConfiguration,
            ...configuration,
            name: `${debuggerType === 'cuda-gdb' ? 'CUDA' : 'C++'}: Debug ${programName}`,
            program: appExecutable(editor),
            internalConsoleOptions: 'openOnSessionStart',
            stopAtEntry: false
        });
        if (!started) {
            vscode.window.showErrorMessage('mem: Could not start the C++ debugger.');
        }
    } catch (e) {
        log('Error debugCpp', e.message);
        vscode.window.showErrorMessage(`mem: ${e.message}`);
    }
}

async function cmdMem() {
    const editor = vscode.window.activeTextEditor;
    if (!editor) {
        log('vscode.window.activeTextEditor; !editor return');
        return;
    }

    if (!await editor.document.save()) return;

    let isPythonFile = true;
    {
        const langId = editor.document.languageId;
        const ext = path.extname(editor.document.uri.fsPath).toLowerCase();
        isPythonFile = langId === 'python' || ext === '.py';
    }
    LOG && log('isPythonFile', isPythonFile);

    const session = vscode.debug.activeDebugSession;
    if (!session) {
        if (isPythonFile){
            log('!session runPython(editor)');
            await runPython(editor);   
        }  
        LOG && log('!session return');         
        return;
    }

    let isPythonProgram = false;
    {
        const session_type = (session.type || '').toLowerCase();
        if (['debugpy', 'python', 'pythonexperimental'].includes(session_type)) {
            isPythonProgram = true;
        }
    }
    LOG && log('isPythonProgram', isPythonProgram);    

    if (!isPythonProgram) { // C++ program
        if (!isPythonFile) { // C++ program, C++ file
            LOG && log('memCppCpp(session) C++ program, C++ file')
            await printStack(session);
            await initialMem(session);

            if (editor.selection.isEmpty) {
                const commandText = editor.document.lineAt(editor.selection.active.line).text;
                if (!/[\r\n]/.test(commandText) && await runExec(commandText)) return;
            }

            await memCppCpp(session);

        } else { // C++ program, Python file
            LOG && log('memCppPython(session) C++ program, Python file')
            await printStack(session);
            await initialMem(session);
            await memCppPython(session);
        }
        
    } else { // Python program
        if (isPythonFile) { // Python program, Python file
            LOG && log('memPythonPython(session) Python program, Python file')
            await printStack(session)
            await memPythonPython(session);
        }
    }
}

function inspectDependencies(source) {
    const clean = source.replace(
        /\/\*[\s\S]*?\*\/|\/\/[^\r\n]*|"(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*'|^[ \t]*#.*(?:\\\r?\n.*)*$/gm,
        text => text.replace(/[^\r\n]/g, ' ')
    );
    const definitions = new Set();
    const references = new Set();
    const keywords = new Set(['if', 'for', 'while', 'switch', 'catch', 'sizeof', 'alignof', 'decltype', 'static_assert']);
    for (const match of clean.matchAll(/\b([A-Za-z_]\w*)\s*(?:<<<[\s\S]*?>>>\s*)?\(/g)) {
        const name = match[1];
        if (keywords.has(name)) continue;
        let end = match.index + match[0].length;
        let depth = 1;
        while (end < clean.length && depth) {
            if (clean[end] === '(') depth++;
            if (clean[end] === ')') depth--;
            end++;
        }
        if (depth) continue;
        // A body distinguishes an implementation from a forward declaration.
        if (/^\s*(?:(?:const|override|final|noexcept)\s*)*\{/.test(clean.slice(end))) {
            definitions.add(name);
        } else {
            references.add(name);
        }
    }
    return { definitions, references };
}

async function findDependencies(document) {
    const folder = vscode.workspace.getWorkspaceFolder(document.uri);
    const root = folder ? folder.uri.fsPath : path.dirname(document.uri.fsPath);
    const uris = await vscode.workspace.findFiles(
        new vscode.RelativePattern(root, '**/*.{cu,cuh,c,cc,cpp,cxx,h,hpp,hxx}'),
        '**/{node_modules,.git,build,dist}/**'
    );
    const files = new Map();
    for (const uri of uris) {
        const candidate = await vscode.workspace.openTextDocument(uri);
        files.set(uri.fsPath, { ...inspectDependencies(candidate.getText()), source: candidate.getText() });
    }
    files.set(document.uri.fsPath, { ...inspectDependencies(document.getText()), source: document.getText() });
    const providers = new Map();
    for (const [filename, info] of files) {
        for (const name of info.definitions) {
            if (!providers.has(name)) providers.set(name, []);
            providers.get(name).push(filename);
        }
    }
    const includedSources = new Set();
    const found = new Set();
    const visited = new Set();
    const pending = [document.uri.fsPath];
    while (pending.length) {
        const filename = pending.pop();
        if (visited.has(filename)) continue;
        visited.add(filename);
        const info = files.get(filename);
        const clean = info.source.replace(/\/\*[\s\S]*?\*\/|\/\/[^\r\n]*/g, ' ');
        for (const include of clean.matchAll(/^[ \t]*#[ \t]*include[ \t]*["<]([^">\r\n]+)[">]/gm)) {
            const local = path.resolve(path.dirname(filename), include[1]);
            const workspacePath = path.resolve(root, include[1]);
            const header = files.has(local) ? local : files.has(workspacePath) ? workspacePath : undefined;
            if (!header) continue; // System/library headers are handled by nvcc.
            pending.push(header);
            if (/\.(cu|c|cc|cpp|cxx)$/i.test(header)) includedSources.add(header);
            if (/\.(h|hpp|hxx|cuh)$/i.test(header)) {
                const stem = header.replace(/\.(h|hpp|hxx|cuh)$/i, '');
                for (const extension of ['.cu', '.cpp', '.cc', '.cxx', '.c']) {
                    const implementation = stem + extension;
                    if (files.has(implementation) && implementation !== document.uri.fsPath) {
                        found.add(implementation);
                        pending.push(implementation);
                    }
                }
            }
        }
        for (const name of info.references) {
            if (info.definitions.has(name)) continue;
            for (const target of providers.get(name) || []) {
                if (target === document.uri.fsPath) continue;
                found.add(target);
                if (!visited.has(target)) pending.push(target);
            }
        }
    }
    return [...found].filter(file => /\.(cu|c|cc|cpp|cxx)$/i.test(file) && !includedSources.has(file)).sort();
}

async function isPythonFile(filename) { return /\.py$/i.test(filename); }
async function isCppFile(filename) { return /\.(c|cc|cpp|h|hpp)$/i.test(filename); }
async function isShaderFile(filename) { return /\.(vert|frag|comp|glsl)$/i.test(filename); }
async function isKernelFile(filename) { return /\.(cu|cuh)$/i.test(filename); }

async function createShaderTest(sourcefile, testfile) {
    const includePath = path.relative(path.dirname(testfile), sourcefile).replace(/\\/g, '/');
    const wrapper = `#include <stdio.h>\n#include "${includePath}"\n\nint main() {\n    printf("hello mingjie!");\n    shader();\n    return 0;\n}\n`;
    try {
        await fs.promises.writeFile(testfile, wrapper, { encoding: 'utf8', flag: 'wx' });
    } catch (e) {
        if (e.code !== 'EEXIST') {
            log('Error creating shader C++ wrapper', e.message);
            vscode.window.showErrorMessage(`mem: ${e.message}`);
            return;
        }
    }  
}

async function createKernelTest(sourcefile, testfile) {
    const includePath = path.relative(path.dirname(testfile), sourcefile).replace(/\\/g, '/');
    const wrapper = `#include <stdio.h>\n#include "kernel.h"\n#include "${includePath}"\n\nint main() {\n    printf("hello mingjie!");\n    return 0;\n}\n`;
    try {
        await fs.promises.writeFile(testfile, wrapper, { encoding: 'utf8', flag: 'wx' });
    } catch (e) {
        if (e.code !== 'EEXIST') {
            log('Error creating kernel C++ wrapper', e.message);
            vscode.window.showErrorMessage(`mem: ${e.message}`);
            return;
        }
    }  
}

async function createCppTest(sourcefile, testfile) {
    const includePath = path.relative(path.dirname(testfile), sourcefile).replace(/\\/g, '/');
    const wrapper = `#include <stdio.h>\n#include "${includePath}"\n\nint main() {\n    printf("hello mingjie!");\n    return 0;\n}\n`;
    try {
        await fs.promises.writeFile(testfile, wrapper, { encoding: 'utf8', flag: 'wx' });
    } catch (e) {
        if (e.code !== 'EEXIST') {
            log('Error creating common C++ wrapper', e.message);
            vscode.window.showErrorMessage(`mem: ${e.message}`);
            return;
        }
    }  
}

function isExistMain(filename) {
    return inspectDependencies(fs.readFileSync(filename, 'utf8')).definitions.has('main');
}

async function cmdDebug() {
    const editor = vscode.window.activeTextEditor;
    if (!editor) {
        log('vscode.window.activeTextEditor; !editor return');
        return;
    }
    
    const filename = editor.document.uri.fsPath;
    if(await isPythonFile(filename)){
        LOG && log('debugPython', 'isPythonFile:true', filename)
        await debugPython(editor);
        return;    
    }

    let sourcefile = "";
    let testfile = "";

    const filename_sub = filename.replace(/_?\.[^./\\]+$/, '');
    const filename_add =  `${filename}_.cc`;

    if (fs.existsSync(filename_sub)){
        sourcefile = filename_sub;
        testfile = filename;
        LOG && log("existsSync sourcefile", sourcefile, testfile)
    } 
    else if (fs.existsSync(filename_add)){
        sourcefile = filename;
        testfile = filename_add;
        LOG && log("existsSync testfile", sourcefile, testfile)
    }
    else {
        if(await isShaderFile(filename)){
            LOG && log("isShaderFile createShaderTest", filename)
            await createShaderTest(filename, filename_add);
        }
        else if(await isKernelFile(filename)){
            LOG && log("isKernelFile createKernelTest", filename)
            await createKernelTest(filename, filename_add);
        }
        else if(!isExistMain(filename)){
            LOG && log("!isExistMain createCppTest", filename)
            await createCppTest(filename, filename_add);
        }
        else{
            LOG && log("buildCpp debugCpp", filename)
            if (!await buildCpp(editor, filename)) return;
            await debugCpp(editor);
            return;
        }
        return;
    }

    if(await isShaderFile(sourcefile)){
        LOG && log("isShaderFile", sourcefile,  "buildShader", testfile)
        if (!await buildShader(editor, testfile)) return;
        await debugCpp(editor);
    }
    else if(await isKernelFile(sourcefile)){
        if (!await vscode.workspace.saveAll()) return;
        if (isExistMain(sourcefile)) {
            LOG && log("isKernelFile", sourcefile, "buildKernelMain", sourcefile)
            if (!await buildKernelMain(editor, sourcefile)) return;
        } else {
            LOG && log("isKernelFile", sourcefile, "buildKernel", testfile)
            if (!await buildKernel(editor, testfile)) return;
        }
        await debugCpp(editor);
    }
    else{
        LOG && log("buildCpp buildCpp", testfile)
        if (!await buildCpp(editor, testfile)) return;
        await debugCpp(editor);
    }
}

async function createPythonScript() {
    const editor = vscode.window.activeTextEditor;
    if (!editor) {
        log('vscode.window.activeTextEditor; !editor return');
        return;
    }
    
    const filename = editor.document.uri.fsPath;
    const isPythonFile = editor.document.languageId === 'python' || /\.py$/i.test(filename);
    const isCppFile = /\.(c|cpp|h|hpp)$/i.test(filename);   
    const isShaderFile = /\.(vert|frag|comp|glsl)$/i.test(filename);
    const isKernelFile = /\.(cu|cuh)$/i.test(filename);

    if(isPythonFile){
        return filename;
    }
    
    const source = `${filename}_.py`;
    const wrapper = "printf('hello mingjie!')\n";
    try {
        await fs.promises.writeFile(source, wrapper, { encoding: 'utf8', flag: 'wx' });
    } catch (e) {
        if (e.code !== 'EEXIST') {
            log('Error creating script wrapper', e.message);
            vscode.window.showErrorMessage(`mem: ${e.message}`);
            return;
        }
    } 
    return "";
}

async function scriptDebug(filename) {
    const folder = vscode.workspace.getWorkspaceFolder(vscode.Uri.file(filename));
    if (!folder) {
        vscode.window.showErrorMessage('mem: Open the project folder before starting the Python script.');
        return;
    }
    const scriptPath = filename.replace(/\\/g, '/');
    await fs.promises.access(scriptPath, fs.constants.R_OK);
    const configurations = vscode.workspace.getConfiguration('launch', folder.uri)
        .get('configurations', []);
    const configuration = configurations.find(config => config.name === 'GDB'
        && config.type === 'cppdbg' && config.request === 'launch'
        && (!config.MIMode || config.MIMode === 'gdb'));
    if (!configuration) {
        vscode.window.showErrorMessage('mem: Add a GDB launch configuration named "GDB" to launch.json.');
        return;
    }

    // Run inside GDB, where the gdb Python module and debug symbols exist.
    // Share GDB's Python globals, including the imported mem helpers.
    const scriptCommand = `python __file__ = ${JSON.stringify(scriptPath)}; exec(compile(open(__file__, 'rb').read(), __file__, 'exec'), globals())`;
    const setupCommands = (configuration.setupCommands || []).filter(command => {
        const match = /^\s*source\s+(.+?)\s*$/.exec(command.text || '');
        const sourcePath = match && match[1].replace(/^"(.*)"$/, '$1')
            .replace(/\$\{workspaceFolder\}/g, folder.uri.fsPath).replace(/\\/g, '/');
        return sourcePath !== scriptPath && command.text !== scriptCommand;
    });
    setupCommands.push({
        text: scriptCommand,
        description: 'Run the selected Python script',
        ignoreFailures: false
    });

    // Start explicitly and await configuration resolution and the build task.
    // This does not depend on the dropdown selection or a temporary provider.
    const started = await vscode.debug.startDebugging(folder, {
        ...configuration,
        name: 'GDB: Python script',
        stopAtEntry: false,
        setupCommands,
        internalConsoleOptions: 'openOnSessionStart'
    });
    if (!started) {
        vscode.window.showErrorMessage('mem: Could not start the Python script.');
    }
}

async function cmdScript() {
    if (vscode.debug.activeDebugSession) {
        vscode.window.showErrorMessage('mem: Stop the current debug session before starting the Python script.');
        return;
    }

    const editor = vscode.window.activeTextEditor;
    if (!editor) return;

    const filename = editor.document.uri.fsPath;
    if(await isPythonFile(filename)){
        LOG && log('scriptDebug', 'isPythonFile:true', filename)
        await scriptDebug(filename);
        return;    
    }

    let sourcefile = "";
    let testfile = "";

    const filename_sub = filename.replace(/_?\.[^./\\]+$/, '');
    const filename_add =  `${filename}_.py`;

    if (fs.existsSync(filename_sub)){
        sourcefile = filename_sub;
        testfile = filename;
    } 
    else if (fs.existsSync(filename_add)){
        sourcefile = filename;
        testfile = filename_add;
    }
    else {
        await createPythonScript();
        return;
    }

    await scriptDebug(testfile);
}

function expandKernelCondition(editor, condition) {
    // Expand global thread coordinates only inside a CUDA kernel body.
    const source = editor.document.getText().slice(0,
        editor.document.offsetAt(editor.selection.active));
    const clean = source.replace(
        /\/\*[\s\S]*?\*\/|\/\/[^\r\n]*|"(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*'/g,
        value => value.replace(/[^\r\n]/g, ' ')
    );
    const inKernel = [...clean.matchAll(/\b__global__\b[^;{}]*\{/g)].some(kernel => {
        let depth = 1;
        for (let i = kernel.index + kernel[0].length; i < clean.length; i++) {
            if (clean[i] === '{') depth++;
            if (clean[i] === '}' && --depth === 0) return false;
        }
        return true;
    });
    if (inKernel) {
        condition = condition.replace(/\bidx\s*\.\s*([xyz])\b/g,
            (_, axis) => `(blockIdx.${axis} * blockDim.${axis} + threadIdx.${axis})`)
            .replace(/(?<![=!<>])=(?!=)/g, '==');
    }
    return condition;
}

async function toggleBreakpoint(){
    const editor = vscode.window.activeTextEditor;
    if (!editor) return;

    let text = editor.document.lineAt(editor.selection.active.line).text;
    text = text.trim().replace(/^(?:\/{2,}|\/\*)\s*/, '')
        .replace(/\s*\*\/\s*$/, '').trim();

    const match = text.match(/^-?exec\s+break(?:\s|$)/);
    if (!match){
        LOG && log("normal breakpoint: editor.debug.action.toggleBreakpoint");
        await vscode.commands.executeCommand('editor.debug.action.toggleBreakpoint');
        return false;
    }

    text = text.slice(match.index).trim();

    if (/^-?exec\s+break(?:\s|$)/.test(text)){
        // -exec break [if] expression (optionally quoted)
        let condition = text.replace(/^-?exec\s+break\s*/, '').trim();
        for (let pass = 0; pass < 2; pass++) {
            if (condition.length >= 2 &&
                ((condition.startsWith("'") && condition.endsWith("'")) ||
                    (condition.startsWith('"') && condition.endsWith('"')))) {
                condition = condition.slice(1, -1).trim();
            }
            if (pass === 0) condition = condition.replace(/^if\s+/, '').trim();
        }

        condition = expandKernelCondition(editor, condition);

        const line = editor.selection.active.line;
        const uri = editor.document.uri;
        const existing = vscode.debug.breakpoints.filter(breakpoint =>
            breakpoint instanceof vscode.SourceBreakpoint &&
            breakpoint.location.uri.toString() === uri.toString() &&
            breakpoint.location.range.start.line === line
        );

        log(`-exec break ${uri.fsPath}:${line + 1}`, condition);

        if (existing.length > 0) {
            vscode.debug.removeBreakpoints(existing);
        } else {
            vscode.debug.addBreakpoints([
                new vscode.SourceBreakpoint(
                    new vscode.Location(uri, new vscode.Position(line, 0)),
                    true, condition || undefined
                )
            ]);
        }
    } 
}

async function activate(context) {
    // NODELETE
    // try {
    //     if (await openingProject(context)) {
    //         return;
    //     }
    // } catch (error) {
    //     console.error('mem: Error opening project', error);
    // }

    try {
        registerShaderDefinitions(vscode, context);
    } catch (error) {
        console.error('mem: Error registering shader definitions', error);
    }

    const memCmd = vscode.commands.registerCommand('mem', async () => { // F8
        try {
            const editor = vscode.window.activeTextEditor;
            if (editor && !editor.document.isClosed && !await editor.document.save()) return;
            await cmdMem();
        } catch (error) {
            console.error('mem: Error running cmdMem', error);
            log('Error running cmdMem', error.message || error);
            vscode.window.showErrorMessage(`mem: ${error.message || error}`);
        }
    });

    const debugCmd = vscode.commands.registerCommand('debug', async () => { // F6
        try {
            const editor = vscode.window.activeTextEditor;
            if (editor && !editor.document.isClosed && !await editor.document.save()) return;
            await cmdDebug();
        } catch (error) {
            console.error('mem: Error running cmdDebug', error);
        }
    });

    const scriptCmd = vscode.commands.registerCommand('script', async () => { // Ctrl+F6
        try {
            const editor = vscode.window.activeTextEditor;
            if (editor && !editor.document.isClosed && !await editor.document.save()) return;
            if (!await vscode.workspace.saveAll()) return;
            await cmdScript();
        } catch (error) {
            console.error('mem: Error running cmdScript', error);
        }
    });

    const stopCmd = vscode.commands.registerCommand('stop', async () => { // Shift+F6
        const editor = vscode.window.activeTextEditor;
        if (editor && !editor.document.isClosed && !await editor.document.save()) return;
        await vscode.commands.executeCommand('workbench.action.debug.stop');
    });

    const toggleBreakpointCmd = vscode.commands.registerCommand('toggleBreakpoint', async () => { // F9
        const editor = vscode.window.activeTextEditor;
        if (editor && !editor.document.isClosed && !await editor.document.save()) return;
        try {
            LOG && log("toggleBreakpoint");
            await toggleBreakpoint();
        } catch (error) {
            vscode.debug.activeDebugConsole.appendLine(`toggleBreakpoint: ${error.message || error}`);
        }
    });

    const stepOverCmd = vscode.commands.registerCommand('stepOver', async () => { // F10
        const editor = vscode.window.activeTextEditor;
        if (editor && !editor.document.isClosed && !await editor.document.save()) return;

        try {
            await vscode.commands.executeCommand('workbench.action.debug.stepOver');
        } catch (error) {
            vscode.debug.activeDebugConsole.appendLine(`stepOver: ${error.message || error}`);
            const session = vscode.debug.activeDebugSession;
            if (session) {
                await aware(session);
                await vscode.commands.executeCommand('workbench.action.debug.stepOver');
            }
        }
    });

    context.subscriptions.push(memCmd, debugCmd, scriptCmd, stopCmd, toggleBreakpointCmd, stepOverCmd, toggleBreakpointCmd);

    // Finish activation before updating configuration files in the background.
    const settingsTask = setImmediate(async () => {
        try {
            await vscodeSettings();
        } catch (error) {
            console.error('mem: Error updating VS Code configuration', error);
        }
    });
    context.subscriptions.push({ dispose: () => clearImmediate(settingsTask) });
}

function deactivate() {}
module.exports = { activate, deactivate };
