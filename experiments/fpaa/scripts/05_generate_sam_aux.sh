#!/usr/bin/env bash
set -euo pipefail

DOMAIN="${1:-Urban}"          # Urban | Rural | Urban,Rural
PYTHON_EXE="${2:-${PYTHON_EXE:-python}}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

DATASET_ROOT="${DATASET_ROOT:-${PROJECT_ROOT}/dataset}"
LOVEDA_ROOT="${LOVEDA_ROOT:-${DATASET_ROOT}/LoveDA/Train}"

export LOVEDA_ROOT
export LOVEDA_DOMAINS="$DOMAIN"

echo "Generate SAM aux maps"
echo "LOVEDA_ROOT   : ${LOVEDA_ROOT}"
echo "LOVEDA_DOMAINS: ${LOVEDA_DOMAINS}"
echo

cd "${PROJECT_ROOT}/SAM_RS"
"$PYTHON_EXE" SAM_utils.py

echo "Done."
