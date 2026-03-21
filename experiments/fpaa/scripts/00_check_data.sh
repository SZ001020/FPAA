#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
DATASET_ROOT="${1:-${DATASET_ROOT:-${PROJECT_ROOT}/../autodl-tmp/dataset}}"

count_images() {
  local path="$1"
  if [[ ! -d "$path" ]]; then
    echo 0
    return
  fi
  find "$path" -type f \( -iname '*.png' -o -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.tif' -o -iname '*.tiff' \) | wc -l
}

targets=(
  "LoveDA/Train"
  "LoveDA/Val"
  "LoveDA/Test"
  "Hunan_Dataset/train"
  "Hunan_Dataset/val"
  "Hunan_Dataset/test"
  "Potsdam"
  "Vaihingen"
)

echo "Dataset root: ${DATASET_ROOT}"
echo "----------------------------------------"
for t in "${targets[@]}"; do
  p="${DATASET_ROOT}/${t}"
  if [[ -d "$p" ]]; then
    exists=true
  else
    exists=false
  fi
  cnt="$(count_images "$p")"
  printf '%-24s exists=%-5s images=%s\n' "$t" "$exists" "$cnt"
done

echo "----------------------------------------"
echo "Data check finished."
