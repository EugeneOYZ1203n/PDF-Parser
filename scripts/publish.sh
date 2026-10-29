#!/bin/bash
rm -rf dist

if [[ -z $1 ]]; then
    echo "No version type specified, bumping patch, building and publishing."
    uv version --bump patch
else
    echo "Bumping $1 version, building and publishing."
    uv version --bump $1
fi
uv build && uv publish --index DPCompute
