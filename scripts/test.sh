#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
exec "${PYTHON:-python}" -m tools.evaluate "$@"
