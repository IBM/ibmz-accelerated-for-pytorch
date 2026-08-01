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
# read-only at /scripts. A user-owned workspace directory is created alongside
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

# Under rootless podman (including when invoked via a docker alias), use the
# cgroupfs cgroup manager to avoid systemd unit errors in SSH sessions, and
# remap workspace ownership into the container's UID namespace.
# Neither applies to real Docker (no podman present).
PODMAN_EXTRA_FLAGS=()
if command -v podman &>/dev/null; then
    PODMAN_EXTRA_FLAGS=(--cgroup-manager=cgroupfs)
fi

echo "Building sample image from ${BASE_IMAGE} ..."
docker build \
    "${PODMAN_EXTRA_FLAGS[@]}" \
    --build-arg BASE_IMAGE="${BASE_IMAGE}" \
    -t "${IMAGE_TAG}" \
    "${SCRIPT_DIR}"

# Create the workspace directory
mkdir -p "${WORKSPACE_DIR}"

if command -v podman &>/dev/null; then
    # Query ibm-user's numeric UID from the built image so podman unshare chown
    # can use it (podman unshare runs inside the user namespace where usernames
    # are not resolved — only numeric UIDs are valid).
    IBM_USER_UID=$(docker run --rm "${PODMAN_EXTRA_FLAGS[@]}" --entrypoint id "${IMAGE_TAG}" -u)
    if [[ -z "${IBM_USER_UID}" ]]; then
        echo "Error: could not determine ibm-user UID from image ${IMAGE_TAG}" >&2
        exit 1
    fi
    podman unshare chown "${IBM_USER_UID}:${IBM_USER_UID}" "${WORKSPACE_DIR}"
fi

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
echo "  python3 /scripts/credit_card_fraud_training.py"
echo ""

docker run -it --rm \
    "${PODMAN_EXTRA_FLAGS[@]}" \
    -v "${SCRIPT_DIR}":/scripts:ro,z \
    -v "${WORKSPACE_DIR}":/workspace:z \
    -w /workspace \
    "${IMAGE_TAG}" \
    bash
