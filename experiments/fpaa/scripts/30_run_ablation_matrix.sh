#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MATRIX_CSV="${1:-${SCRIPT_DIR}/../configs/ablation_matrix.csv}"
PYTHON_EXE="${2:-${PYTHON_EXE:-python}}"

if [[ ! -f "$MATRIX_CSV" ]]; then
  echo "Matrix file not found: $MATRIX_CSV"
  exit 1
fi

row_count=$(($(wc -l < "$MATRIX_CSV") - 1))
if (( row_count <= 0 )); then
  echo "Matrix file is empty: $MATRIX_CSV"
  exit 1
fi

echo "Loaded matrix rows: $row_count"

is_header=true
while IFS=',' read -r run_name backbone direction seed quality_mode use_lite_da; do
  if $is_header; then
    is_header=false
    continue
  fi

  if [[ -z "${run_name}" || "${run_name:0:1}" == "#" ]]; then
    continue
  fi

  echo "----------------------------------------"
  echo "Run: ${run_name}"
  echo "backbone=${backbone} direction=${direction} seed=${seed} quality_mode=${quality_mode} use_lite_da=${use_lite_da}"

  export FPAA_QUALITY_MODE="${quality_mode}"
  export FPAA_USE_LITE_DA="${use_lite_da}"

  if [[ "${backbone}" == "GLGAN" ]]; then
    bash "${SCRIPT_DIR}/20_run_glgan_baseline.sh" "${direction}" "${seed}" "${PYTHON_EXE}"
  else
    bash "${SCRIPT_DIR}/10_run_fm_baseline.sh" "${backbone}" "${direction}" "${seed}" "${PYTHON_EXE}"
  fi
done < "$MATRIX_CSV"

echo "----------------------------------------"
echo "All matrix runs finished."
