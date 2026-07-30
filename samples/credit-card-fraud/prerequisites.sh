#!/bin/bash
# Prerequisites for the credit-card-fraud sample.
#
# Usage:
#   ./prerequisites.sh <base-image> [/path/to/card_transaction.v1.csv]
#
# <base-image> must be an IBM Z Accelerated for PyTorch production image, e.g.:
#   icr.io/ibmz/ibmz-accelerated-for-pytorch:1.5.0
#
# The optional second argument is the path to card_transaction.v1.csv. If
# provided, the file is copied into the workspace directory automatically.
#
# The script builds a new container image with all dependencies pre-installed,
# then starts an interactive shell inside it. Sample scripts are mounted
# read-only at /sample. A user-owned workspace directory is created alongside
# the sample scripts and mounted at /workspace — this is where output files
# (model checkpoints, test data, etc.) will be written.

set -euo pipefail

BASE_IMAGE="${1:-}"
if [[ -z "${BASE_IMAGE}" ]]; then
    echo "Error: base image argument is required." >&2
    echo "Usage: ./prerequisites.sh <base-image> [/path/to/card_transaction.v1.csv]" >&2
    exit 1
fi

CSV_PATH="${2:-}"
CSV_FILENAME="card_transaction.v1.csv"

if ! command -v docker &>/dev/null; then
    echo "Error: docker not found. Run this script on the host, not inside a container." >&2
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_DIR="${SCRIPT_DIR}/workspace"
IMAGE_TAG="ccf-sample:latest"

echo "Building sample image from ${BASE_IMAGE} ..."
docker build \
    --build-arg BASE_IMAGE="${BASE_IMAGE}" \
    -t "${IMAGE_TAG}" \
    "${SCRIPT_DIR}"

# Create the workspace directory
mkdir -p "${WORKSPACE_DIR}"

# Resolve the dataset location:
#   1. Explicit path from $2 argument
#   2. Autodetect in the script directory (credit-card-fraud/)
#   3. Already present in workspace/ — nothing to do
#   4. Not found — warn and continue, user must add it manually
if [[ -n "${CSV_PATH}" ]]; then
    if [[ ! -f "${CSV_PATH}" ]]; then
        echo "Error: CSV file not found: ${CSV_PATH}" >&2
        exit 1
    fi
    echo "Copying ${CSV_FILENAME} to ${WORKSPACE_DIR} ..."
    cp "${CSV_PATH}" "${WORKSPACE_DIR}/${CSV_FILENAME}"
elif [[ -f "${WORKSPACE_DIR}/${CSV_FILENAME}" ]]; then
    echo "${CSV_FILENAME} already present in workspace, skipping copy."
elif [[ -f "${SCRIPT_DIR}/${CSV_FILENAME}" ]]; then
    echo "Found ${CSV_FILENAME} in sample directory, copying to ${WORKSPACE_DIR} ..."
    cp "${SCRIPT_DIR}/${CSV_FILENAME}" "${WORKSPACE_DIR}/${CSV_FILENAME}"
else
    CSV_PATH=""  # ensure the warning block below fires
fi

echo ""
echo "Build complete."
echo ""
if [[ -z "${CSV_PATH}" && ! -f "${WORKSPACE_DIR}/${CSV_FILENAME}" ]]; then
    echo "Warning: ${CSV_FILENAME} was not found automatically."
    echo "Before running the sample, place it in:"
    echo "  ${WORKSPACE_DIR}"
    echo ""
fi
echo "Inside the container, run scripts from /workspace, e.g.:"
echo "  python3 /sample/credit_card_fraud_training.py"
echo ""

docker run -it --rm \
    -v "${SCRIPT_DIR}":/sample:ro,z \
    -v "${WORKSPACE_DIR}":/workspace:z \
    -w /workspace \
    "${IMAGE_TAG}" \
    bash
