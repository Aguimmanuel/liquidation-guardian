#!/usr/bin/env bash
# Start the Liquidation Guardian web app.
# Usage: ./run.sh   (Linux / macOS)
set -e
cd "$(dirname "$0")"

if ! command -v python3 >/dev/null 2>&1; then
  echo "python3 not found — install Python 3.10+ first."
  exit 1
fi

python3 -m pip install -r requirements.txt
python3 -m uvicorn app.main:app --host 0.0.0.0 --port 8000
