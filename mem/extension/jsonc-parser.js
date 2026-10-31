'use strict';

// Local JSON-with-comments reader and targeted editor for settings.js.
// Supports parse, modify (existing paths or new object properties), and applyEdits.
function read(text, options = {}) {
    let offset = 0;
    function fail(message) {
        const error = new SyntaxError(message);
        error.offset = offset;
        throw error;
    }
    function skip() {
        while (offset < text.length) {
            if (/\s/.test(text[offset])) { offset++; continue; }
            if (text.startsWith('//', offset)) {
                if (options.disallowComments) fail('Comments are disabled');
                while (offset < text.length && !/[\r\n]/.test(text[offset])) offset++;
                continue;
            }
            if (text.startsWith('/*', offset)) {
                if (options.disallowComments) fail('Comments are disabled');
                const end = text.indexOf('*/', offset + 2);
                if (end < 0) fail('Unterminated comment');
                offset = end + 2;
                continue;
            }
            break;
        }
    }
    function string() {
        const start = offset++;
        while (offset < text.length) {
            const character = text[offset++];
            if (character === '\\') { offset++; continue; }
            if (character === '"') return JSON.parse(text.slice(start, offset));
        }
        fail('Unterminated string');
    }
    function value() {
        skip();
        const start = offset;
        const opening = text[offset];
        if (opening === '{' || opening === '[') {
            offset++;
            const object = opening === '{';
            const closing = object ? '}' : ']';
            const result = object ? {} : [];
            const children = [];
            let trailingComma = false;
            skip();
            while (text[offset] !== closing) {
                let key = children.length;
                if (object) {
                    if (text[offset] !== '"') fail('Expected property name');
                    key = string();
                    skip();
                    if (text[offset++] !== ':') fail('Expected colon');
                }
                const child = value();
                children.push({ key, node: child });
                // Define properties explicitly so __proto__ remains ordinary JSON data.
                Object.defineProperty(result, key, {
                    value: child.value, enumerable: true, writable: true, configurable: true
                });
                skip();
                if (text[offset] !== ',') break;
                offset++;
                skip();
                trailingComma = text[offset] === closing;
                if (trailingComma && !options.allowTrailingComma) fail('Trailing comma');
            }
            const close = offset;
            if (text[offset++] !== closing) fail('Expected closing bracket');
            return { start, end: offset, close, value: result, children, object, trailingComma };
        }
        let result;
        if (opening === '"') result = string();
        else {
            const match = /^(?:true|false|null|-?(?:0|[1-9]\d*)(?:\.\d+)?(?:[eE][+-]?\d+)?)/.exec(text.slice(offset));
            if (!match) fail('Expected JSON value');
            offset += match[0].length;
            result = JSON.parse(match[0]);
        }
        return { start, end: offset, value: result };
    }
    const root = value();
    skip();
    if (offset !== text.length) fail('Unexpected content after JSON value');
    return root;
}

function parse(text, errors = [], options = {}) {
    try { return read(text, options).value; }
    catch (error) {
        errors.push({ error: error.message, offset: error.offset ?? 0, length: 1 });
        return undefined;
    }
}

function modify(text, path, value, options = {}) {
    if (value === undefined) throw new Error('Deleting JSON properties is not supported');
    const format = options.formattingOptions || {};
    const eol = format.eol || (text.includes('\r\n') ? '\r\n' : '\n');
    const indent = format.insertSpaces === false ? '\t' : ' '.repeat(format.tabSize ?? 4);
    function lineIndent(position) {
        const start = text.lastIndexOf('\n', position - 1) + 1;
        return /^[\t ]*/.exec(text.slice(start, position))[0];
    }
    function serialize(base) {
        return JSON.stringify(value, null, indent).replace(/\n/g, eol + base);
    }
    let node = read(text, { allowTrailingComma: true });
    if (!path.length) return [{ offset: node.start, length: node.end - node.start, content: serialize('') }];
    for (let index = 0; index < path.length; index++) {
        const key = path[index];
        if (!node.children || (node.object ? typeof key !== 'string' : !Number.isInteger(key))) {
            throw new Error('Invalid JSON edit path');
        }
        const child = [...node.children].reverse().find(entry => entry.key === key);
        if (child) {
            node = child.node;
            if (index === path.length - 1) {
                return [{ offset: node.start, length: node.end - node.start, content: serialize(lineIndent(node.start)) }];
            }
            continue;
        }
        if (!node.object || index !== path.length - 1) {
            throw new Error('JSON edit parent does not exist');
        }
        const base = lineIndent(node.start);
        const property = JSON.stringify(key) + ': ' + serialize(base + indent);
        // Insert before the closing brace without moving existing comments or values.
        const edits = [];
        const last = node.children.at(-1)?.node;
        if (last && !node.trailingComma) edits.push({ offset: last.end, length: 0, content: ',' });
        const lineStart = text.lastIndexOf('\n', node.close - 1) + 1;
        const closingOnOwnLine = lineStart > node.start && /^[\t ]*$/.test(text.slice(lineStart, node.close));
        const insertion = closingOnOwnLine ? lineStart : node.close;
        edits.push({ offset: insertion, length: 0,
            content: (closingOnOwnLine ? '' : eol) + base + indent + property +
                (node.trailingComma ? ',' : '') + eol + (closingOnOwnLine ? '' : base) });
        if (edits.length === 2 && edits[0].offset === edits[1].offset) {
            edits[1].content = edits[0].content + edits[1].content;
            edits.shift();
        }
        return edits;
    }
    return [];
}

function applyEdits(text, edits) {
    let boundary = text.length;
    for (const edit of [...edits].sort((a, b) => b.offset - a.offset)) {
        if (edit.offset < 0 || edit.length < 0 || edit.offset + edit.length > boundary) {
            throw new Error('Invalid or overlapping JSON edits');
        }
        text = text.slice(0, edit.offset) + edit.content + text.slice(edit.offset + edit.length);
        boundary = edit.offset;
    }
    return text;
}

module.exports = { parse, modify, applyEdits };
