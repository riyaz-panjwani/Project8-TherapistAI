#!/bin/bash
# Run from anywhere — always finds the backend relative to this script
DIR="$(cd "$(dirname "$0")" && pwd)"

# Activate venv if it exists (use: python3.12 -m venv .venv to create it)
if [ -f "$DIR/.venv/bin/activate" ]; then
  source "$DIR/.venv/bin/activate"
fi

cd "$DIR/backend"
echo "Starting TherapistAI at http://127.0.0.1:8001"
python3 -m uvicorn main:app --host 0.0.0.0 --port 8001 --reload
