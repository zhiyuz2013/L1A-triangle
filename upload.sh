#!/usr/bin/env bash
set -euo pipefail

# Keep Git output in the terminal so the upload never pauses in a pager.
export GIT_PAGER=cat

project_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repository_url='https://github.com/zhiyuz2013/L1A-triangle.git'
temp_dir=$(mktemp -d "${TMPDIR:-/tmp}/l1a-upload.XXXXXXXX")
trap 'rm -rf -- "$temp_dir"' EXIT

echo 'Downloading the current main branch from GitHub...'
echo 'If prompted for a password, enter a GitHub personal access token.'
git -c credential.username=zhiyuz2013 clone --branch main --single-branch \
    "$repository_url" "$temp_dir/repository"

echo 'Copying project files (excluding generated build files and caches)...'
tar -C "$project_dir" \
    --exclude='./.git' --exclude='./.agents' --exclude='./.codex' --exclude='./.aws' \
    --exclude='./build' --exclude='./CMakeFiles' \
    --exclude='__pycache__' --exclude='*.pyc' \
    -cf - . | tar -C "$temp_dir/repository" -xf -

cd "$temp_dir/repository"
git add -A
if git diff --cached --quiet; then
    echo 'The repository already contains these files. Nothing to upload.'
    exit 0
fi

if ! git var GIT_AUTHOR_IDENT >/dev/null 2>&1; then
    read -r -p 'Git author name: ' author_name
    read -r -p 'Git author email: ' author_email
    git config user.name "$author_name"
    git config user.email "$author_email"
fi

echo 'Files to be uploaded:'
git diff --cached --stat
git commit -m 'Upload project files'
echo 'Sending the commit to GitHub.'
echo 'When prompted for a password in this terminal, paste your GitHub personal access token.'
# VS Code can set an askpass helper that waits for a hidden dialog. Request
# credentials in the terminal instead and do not save the token.
unset GIT_ASKPASS SSH_ASKPASS
git -c core.askPass= -c credential.helper= -c credential.username=zhiyuz2013 \
    push --progress origin HEAD:main
echo 'Upload complete: https://github.com/zhiyuz2013/L1A-triangle'
