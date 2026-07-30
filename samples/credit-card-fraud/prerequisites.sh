#!/bin/bash
# Prerequisites for the credit-card-fraud sample.
#
# Usage:
#   ./prerequisites.sh <base-image> [/path/to/card_transaction.v1.csv]
#
# <base-image> must be an IBM Z Accelerated for PyTorch production image, e.g.:
#   icr.io/zai_pytorch/v1.5.0_3q26/prod_pt-2.11_cp-3.12_ubi-10.2:zosdev_pt_v1.5.0_3q26-rc4
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

if ! command -v podman &>/dev/null; then
    echo "Error: podman not found. Run this script on the host, not inside a container." >&2
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_DIR="${SCRIPT_DIR}/workspace"
IMAGE_TAG="ccf-sample:latest"

echo "Building sample image from ${BASE_IMAGE} ..."
podman build \
    --build-arg BASE_IMAGE="${BASE_IMAGE}" \
    -t "${IMAGE_TAG}" \
    "${SCRIPT_DIR}"

# Query ibm-user's numeric UID from the built image so podman unshare chown
# can use it (podman unshare runs inside the user namespace where usernames
# are not resolved — only numeric UIDs are valid).
IBM_USER_UID=$(podman run --rm --entrypoint id "${IMAGE_TAG}" -u)
if [[ -z "${IBM_USER_UID}" ]]; then
    echo "Error: could not determine ibm-user UID from image ${IMAGE_TAG}" >&2
    exit 1
fi

# Create the workspace directory and transfer ownership to ibm-user's UID
# within the rootless UID namespace, so the container can write output files
mkdir -p "${WORKSPACE_DIR}"
podman unshare chown "${IBM_USER_UID}:${IBM_USER_UID}" "${WORKSPACE_DIR}"

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

podman run -it --rm \
    --device /dev/vfio \
    -v "${SCRIPT_DIR}":/sample:ro,z \
    -v "${WORKSPACE_DIR}":/workspace:z \
    -w /workspace \
    "${IMAGE_TAG}" \
    bash
