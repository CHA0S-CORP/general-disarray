#!/usr/bin/env sh
# Dockerized build for n8n-nodes-general-disarray — no host node/npm needed.
# Works on arm64 (DGX Spark / GB10); output lands in ./dist owned by the host user.
set -eu
cd "$(dirname "$0")"
docker run --rm \
  -u "$(id -u):$(id -g)" \
  -e HOME=/tmp -e npm_config_cache=/tmp/.npm \
  -v "$PWD":/app -w /app \
  node:20-alpine \
  sh -c "npm install --no-audit --no-fund --ignore-scripts && npm run build"
echo "Built:"
ls dist/nodes/*/*.node.js dist/credentials/*.credentials.js
