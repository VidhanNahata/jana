#!/bin/sh
# Assemble the Render deploy repo contents into $1 (backend code + Docker files only)
set -e
out="$1"
root="$(cd "$(dirname "$0")/.." && pwd)"
mkdir -p "$out"
# Keep $out/.git so the staged dir can be re-used as the deploy repo checkout
find "$out" -mindepth 1 -maxdepth 1 ! -name .git -exec rm -rf {} +
cp "$root"/deploy/render/Dockerfile "$root"/deploy/render/README.md "$root"/deploy/render/requirements.txt "$out"/
rsync -a --exclude '__pycache__' --exclude '*.pt' --exclude 'tests' "$root"/backend "$out"/
printf '__pycache__/\n*.pyc\n' > "$out"/.gitignore
