<!-- markdownlint-disable MD033 -->

# MNIST Sample

This sample uses the `python-mnist` package to load the MNIST data set,
trains a model on CPU, then runs inference on the NNPA device. It runs
directly in the base container — no `prerequisites.sh` is needed.

> If you are using rootless podman, see the
> [Running with podman](../README.md#running-with-podman) section in the
> top-level samples README before proceeding.

## Running the Sample

Run these commands from the **host** machine. Replace `X.X.X` with the
current version of the container image.

First, start an interactive container shell with a named volume for the
workspace:

```bash
docker run -it --rm \
    -v "$(pwd)":/scripts:ro,z \
    -v mnist-workspace:/workspace \
    -w /workspace \
    icr.io/ibmz/ibmz-accelerated-for-pytorch:X.X.X bash
```

## Prerequisites

Inside the container, install the `python-mnist` package:

```bash
pip install python-mnist
```

## Copying the Data Set into the Workspace

The training script looks for the MNIST data files in a `./data` sub-directory
of the current working directory (`/workspace`). Open a **second terminal on
the host** and copy the bundled data directory into the running container:

```bash
# Find the running container ID
docker ps

# Copy the data directory
docker cp data/ <container-id>:/workspace/
```

Then return to the container shell before running training.

## Training

Training always runs on CPU. Train the model and save it to disk. You can
specify the number of epochs with `--epochs` (default: `14`):

```bash
python /scripts/mnist_training.py --epochs 2 --save-model
```

## Inference on NNPA

After training with `--save-model`, run inference on the NNPA device:

```bash
python /scripts/mnist_infer.py
```

## Inference on CPU

To run inference on the CPU instead, pass `--no-nnpa`:

```bash
python /scripts/mnist_infer.py --no-nnpa
```

## Cleanup

When you are finished with the sample, remove the stopped container and the
workspace volume:

```bash
docker container prune -f
docker volume rm mnist-workspace
```

## Known Issues

There are no known open issues with this sample.
