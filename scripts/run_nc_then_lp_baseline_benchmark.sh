#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"
OUTPUT_ROOT="${BENCHMARK_OUTPUT_ROOT:-outputs/baseline_benchmark}"
DEVICE="cuda:0"
RESUME_ARGS=()
while [[ "$#" -gt 0 ]]; do
    case "$1" in
        --device)
            if [[ "$#" -lt 2 ]]; then
                echo "Missing value for --device" >&2
                echo "Usage: $0 [--device DEVICE] [--resume]" >&2
                exit 2
            fi
            DEVICE="$2"
            shift 2
            ;;
        --resume)
            RESUME_ARGS+=(--resume)
            shift
            ;;
        *)
            echo "Unknown argument: $1" >&2
            echo "Usage: $0 [--device DEVICE] [--resume]" >&2
            exit 2
            ;;
    esac
done

echo "Starting NC benchmark on ${DEVICE} (Movies, Grocery, Toys, Reddit-S, ele-fashion)."
"${PYTHON_BIN}" scripts/run_nc_baseline_benchmark.py \
    --device "${DEVICE}" \
    --output-root "${OUTPUT_ROOT}" \
    "${RESUME_ARGS[@]}"

echo "NC benchmark completed. Starting LP benchmark on ${DEVICE} (Sports, Cloth)."
"${PYTHON_BIN}" scripts/run_lp_baseline_benchmark.py \
    --device "${DEVICE}" \
    --output-root "${OUTPUT_ROOT}" \
    "${RESUME_ARGS[@]}"

echo "NC and LP baseline benchmarks completed. Results: ${OUTPUT_ROOT}"
