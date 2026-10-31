#!/usr/bin/env bash
set -euo pipefail

cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

vsix="mem-$(node -p 'require("./package.json").version').vsix"
vsce package --allow-missing-repository --skip-license --out "$vsix"
code --install-extension "$vsix" --force

