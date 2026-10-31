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

// Lightweight source navigation for GLSL files, independent of their language mode.
const shaderExtensions = ['vert', 'frag', 'geom', 'tesc', 'tese', 'comp', 'glsl'];

function definitions(source) {
    // Keep offsets intact while excluding comments, strings, and directives.
    const clean = source.replace(/\/\*[\s\S]*?\*\/|\/\/[^\r\n]*|"(?:\\.|[^"\\])*"|^[ \t]*#.*$/gm,
        text => text.replace(/[^\r\n]/g, ' '));
    const tokens = Array.from(clean.matchAll(/[A-Za-z_]\w*|[^\s]/g), m => ({ text: m[0], offset: m.index }));
    const root = { start: 0, end: source.length, parent: null };
    const stack = [root];
    const pairs = new Map();
    const opens = [];
    tokens.forEach((token, index) => {
        token.scope = stack[stack.length - 1];
        if (token.text === '{') {
            token.body = { start: token.offset, end: source.length, parent: token.scope };
            stack.push(token.body);
        } else if (token.text === '}' && stack.length > 1) {
            stack.pop().end = token.offset;
        }
        if (['(', '[', '{'].includes(token.text)) opens.push(index);
        if ([')', ']', '}'].includes(token.text)) {
            const open = opens.pop();
            if (open !== undefined) pairs.set(open, index);
        }
    });
    const types = new Set(['void', 'bool', 'int', 'uint', 'float', 'double']);
    for (const token of tokens) {
        if (/^(?:[biud]?vec[234]|d?mat[234](?:x[234])?|[iu]?(?:sampler|image)\w+)$/.test(token.text)) types.add(token.text);
    }
    tokens.forEach((token, i) => {
        if (token.text === 'struct' && tokens[i + 1]) types.add(tokens[i + 1].text);
    });
    const result = [];
    const add = (token, scope = token.scope) => result.push({ name: token.text, offset: token.offset, scope });
    tokens.forEach((token, i) => {
        if (token.text === '#') return;
        // Uniform block instances: uniform Surface { ... } surface;
        if (token.text === '{' && ['uniform', 'buffer'].includes(tokens[i - 2]?.text)) {
            add(tokens[i - 1]);
            const instance = tokens[pairs.get(i) + 1];
            if (instance && /^[A-Za-z_]\w*$/.test(instance.text)) {
                add(instance, token.scope);
                token.body.instance = instance.text;
            }
        }
        if (!types.has(token.text)) return;
        const name = tokens[i + 1];
        if (!name || !/^[A-Za-z_]\w*$/.test(name.text)) return;
        const next = tokens[i + 2];
        if (!next) return;
        if (next.text === '(') {
            const close = pairs.get(i + 2);
            const body = tokens[close + 1];
            if (!body || !['{', ';'].includes(body.text)) return;
            add(name);
            if (body.body) {
                // Function parameters belong to the function body, not the global scope.
                for (let j = i + 3; j < close; j++) tokens[j].scope = body.body;
            }
        } else if (['=', ';', ',', '[', ')'].includes(next.text)) {
            add(name);
            // Additional declarators, ignoring commas in initializers and arrays.
            let j = i + 2;
            while (j < tokens.length && ![';', ')', '{', '}'].includes(tokens[j].text)) {
                if (['(', '['].includes(tokens[j].text) && pairs.has(j)) j = pairs.get(j);
                else if (tokens[j].text === ',' && /^[A-Za-z_]\w*$/.test(tokens[j + 1]?.text || '')
                    && ['=', ';', ',', '['].includes(tokens[j + 2]?.text)) add(tokens[j + 1]);
                j++;
            }
        }
    });
    return result;
}

function findDefinition(source, name, offset) {
    const wordStart = offset - (source.slice(0, offset).match(/[A-Za-z_0-9]*$/)?.[0].length || 0);
    const owner = source.slice(0, wordStart).match(/([A-Za-z_]\w*)\s*\.\s*$/)?.[1];
    const candidates = definitions(source).filter(item => item.name === name
        && (owner ? item.scope.instance === owner : item.scope.start <= offset && offset <= item.scope.end)
        && (item.scope.parent === null || item.offset <= offset));
    candidates.sort((a, b) => b.scope.start - a.scope.start
        || Number(b.offset <= offset) - Number(a.offset <= offset)
        || (a.offset <= offset && b.offset <= offset ? b.offset - a.offset : a.offset - b.offset));
    return candidates[0]?.offset;
}

function registerShaderDefinitions(vscode, context) {
    const path = require('path');
    context.subscriptions.push(vscode.languages.registerDefinitionProvider(
        shaderExtensions.map(ext => ({ scheme: 'file', pattern: `**/*.${ext}` })),
        {
            async provideDefinition(document, position, cancellation) {
                const line = document.lineAt(position.line).text;
                const include = /^[ \t]*#[ \t]*include[ \t]*"([^"]+)"/.exec(line);
                if (include) {
                    const start = line.indexOf('"') + 1;
                    if (position.character >= start && position.character <= start + include[1].length) {
                        const uri = vscode.Uri.file(path.resolve(path.dirname(document.uri.fsPath), include[1]));
                        try {
                            await vscode.workspace.fs.stat(uri);
                            return new vscode.Location(uri, new vscode.Position(0, 0));
                        } catch { return undefined; }
                    }
                }
                if (cancellation.isCancellationRequested) return undefined;
                const range = document.getWordRangeAtPosition(position);
                if (!range) return undefined;
                const offset = findDefinition(document.getText(), document.getText(range), document.offsetAt(position));
                if (offset === undefined) return undefined;
                const start = document.positionAt(offset);
                return new vscode.Location(document.uri, new vscode.Range(start, start.translate(0, document.getText(range).length)));
            }
        }
    ));
}

module.exports = { findDefinition, registerShaderDefinitions };
