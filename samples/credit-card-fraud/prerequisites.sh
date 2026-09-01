#!/bin/bash
# Prerequisites for the credit-card-fraud sample.
#
# Usage:
#   ./prerequisites.sh <base-image>
#
# <base-image> must be an IBM Z Accelerated for PyTorch production image, e.g.:
#   icr.io/ibmz/ibmz-accelerated-for-pytorch:1.5.0
#
# The script builds a new container image with all dependencies pre-installed.
# A timestamped image tag is generated so re-running the script never
# overwrites a previously built image.
#
# After the build, the image tag and an example docker run command are printed.
# See the sample README for full instructions on running the sample.
#
# When you are finished with the sample, remove the workspace volume with:
#   docker volume rm pytorch-ccf-workspace

set -euo pipefail

BASE_IMAGE="${1:-}"
if [[ -z "${BASE_IMAGE}" ]]; then
    echo "Error: base image argument is required." >&2
    echo "Usage: ./prerequisites.sh <base-image>" >&2
    exit 1
fi

if ! command -v docker &>/dev/null; then
    echo "Error: docker not found. Run this script on the host, not inside a container." >&2
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TIMESTAMP="$(date +%Y%m%d-%H%M%S)"
IMAGE_TAG="pytorch-ccf-sample:${TIMESTAMP}"
VOLUME_NAME="pytorch-ccf-workspace"

echo "Building sample image from ${BASE_IMAGE} ..."
docker build \
    --build-arg BASE_IMAGE="${BASE_IMAGE}" \
    -t "${IMAGE_TAG}" \
    "${SCRIPT_DIR}"

echo ""
echo "Build complete."
echo ""
echo "Image tag: ${IMAGE_TAG}"
echo ""
echo "To start the sample container, run:"
echo ""
echo "  docker run -it --rm \\"
echo "      -v ${SCRIPT_DIR}:/scripts:ro,z \\"
echo "      -v ${VOLUME_NAME}:/workspace \\"
echo "      -w /workspace \\"
echo "      ${IMAGE_TAG} \\"
echo "      bash"
echo ""
echo "See the sample README for instructions on copying the dataset into the"
echo "container before running the sample."
echo ""
echo "When finished, remove the workspace volume with:"
echo "  docker volume rm ${VOLUME_NAME}"
echo ""
echo "If using rootless Podman, run 'docker container prune -f' after exiting"
echo "the container to ensure networking processes are cleaned up."
