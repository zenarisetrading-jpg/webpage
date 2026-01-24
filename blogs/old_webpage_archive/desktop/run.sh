#!/bin/bash
# Run Saddle AdPulse Desktop App
# This script ensures the correct Python environment is used

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV_DIR="$SCRIPT_DIR/../.venv"

# Always use venv's streamlit
exec "$VENV_DIR/bin/streamlit" run "$SCRIPT_DIR/ppcsuite_v4_ui_experiment.py" "$@"
