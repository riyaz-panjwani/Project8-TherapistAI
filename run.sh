#!/bin/bash
# Run from anywhere — always finds the backend relative to this script
DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$DIR/backend"

# Set your Anthropic API key here (or export it in your shell profile)
# export ANTHROPIC_API_KEY="sk-ant-..."

echo "Starting TherapistAI at http://127.0.0.1:8001"
python3 -m uvicorn main:app --host 0.0.0.0 --port 8001 --reload
