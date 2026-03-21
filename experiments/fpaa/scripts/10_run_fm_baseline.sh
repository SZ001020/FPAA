#!/usr/bin/env bash
set -euo pipefail

BACKBONE="${1:-MFNet}"
DIRECTION="${2:-R2U}"
SEED="${3:-3407}"
PYTHON_EXE="${4:-${PYTHON_EXE:-python}}"

if [[ "$BACKBONE" != "MFNet" && "$BACKBONE" != "SAM_RS" ]]; then
  echo "Invalid BACKBONE: $BACKBONE (allowed: MFNet, SAM_RS)"
  exit 1
fi

if [[ "$DIRECTION" != "R2U" && "$DIRECTION" != "U2R" ]]; then
  echo "Invalid DIRECTION: $DIRECTION (allowed: R2U, U2R)"
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
EXP_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
LOG_DIR="${EXP_ROOT}/logs"
mkdir -p "$LOG_DIR"

RUN_ID="LoveDA_${DIRECTION}_${BACKBONE}_s${SEED}_$(date +%Y%m%d_%H%M%S)"
LOG_PATH="${LOG_DIR}/${RUN_ID}.log"

export PYTHONHASHSEED="$SEED"
export EXP_DIRECTION="$DIRECTION"
export EXP_SEED="$SEED"

if [[ "$BACKBONE" == "MFNet" ]]; then
  ENTRY="${PROJECT_ROOT}/MFNet/train.py"
else
  ENTRY="${PROJECT_ROOT}/SAM_RS/train.py"
fi

echo "Run ID: $RUN_ID"
echo "Entry : $ENTRY"
echo "Log   : $LOG_PATH"
echo
echo "提示: 当前仓库训练入口多为脚本内配置，请先确认对应 utils/train 中 DATASET 与数据根目录设置。"
echo "开始执行..."

"$PYTHON_EXE" "$ENTRY" 2>&1 | tee "$LOG_PATH"

echo "Finished: $RUN_ID"
