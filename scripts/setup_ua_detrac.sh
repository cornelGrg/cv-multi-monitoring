#!/usr/bin/env bash
# Run from the repository root, with the Python environment activated.
set -euo pipefail
python -m scripts.prepare_ua_detrac "$@"
