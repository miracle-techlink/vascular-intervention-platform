#!/bin/bash
# Build frontend → frontend/dist/
set -e
cd "$(dirname "$0")/frontend"
npm install
npm run build
echo "✓ frontend built to frontend/dist/"
