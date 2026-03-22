#!/usr/bin/env bash
set -euo pipefail

DIRECTION="${1:-R2U}"
SEED="${2:-3407}"
PYTHON_EXE="${3:-${PYTHON_EXE:-python}}"

if [[ "$DIRECTION" != "R2U" && "$DIRECTION" != "U2R" ]]; then
  echo "Invalid DIRECTION: $DIRECTION (allowed: R2U, U2R)"
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
EXP_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
LOG_DIR="${EXP_ROOT}/logs"
mkdir -p "$LOG_DIR"

DATASET_ROOT="${DATASET_ROOT:-${PROJECT_ROOT}/dataset}"
GLGAN_LOVEDA_ROOT="${GLGAN_LOVEDA_ROOT:-${DATASET_ROOT}/LoveDA/Train}"
export DATASET_ROOT
export GLGAN_LOVEDA_ROOT

RUN_ID="LoveDA_${DIRECTION}_GLGAN_s${SEED}_$(date +%Y%m%d_%H%M%S)"
LOG_PATH="${LOG_DIR}/${RUN_ID}.log"

export PYTHONHASHSEED="$SEED"
export EXP_DIRECTION="$DIRECTION"
export EXP_SEED="$SEED"

if [[ "$DIRECTION" == "R2U" ]]; then
  ENTRY="${PROJECT_ROOT}/GLGAN/GLGAN_LoveDA_R2U.py"
else
  ENTRY="${PROJECT_ROOT}/GLGAN/GLGAN_LoveDA_U2R.py"
fi

echo "Run ID: $RUN_ID"
echo "Entry : $ENTRY"
echo "Log   : $LOG_PATH"
echo
echo "提示: 请先确认 GLGAN 脚本中的数据目录与预训练权重路径。"
echo "开始执行..."

"$PYTHON_EXE" "$ENTRY" 2>&1 | tee "$LOG_PATH"

echo "Finished: $RUN_ID"
