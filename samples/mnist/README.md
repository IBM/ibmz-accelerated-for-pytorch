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

First, create a workspace directory and start an interactive container shell:

```bash
mkdir -p workspace

docker run -it --rm \
    -v "$(pwd)":/sample:ro,z \
    -v "$(pwd)/workspace":/workspace:z \
    -w /workspace \
    icr.io/ibmz/ibmz-accelerated-for-pytorch:X.X.X bash
```

## Prerequisites

Inside the container, install the `python-mnist` package:

```bash
pip install python-mnist
```

## Training on CPU

Train the model and save it to disk. You can specify the number of epochs
with `--epochs`.

```bash
python /sample/mnist_training.py --epochs 2 --save-model
```

## Inference on NNPA Device

After training with `--save-model`, run inference on the NNPA device:

```bash
python /sample/mnist_infer.py
```

## Inference on CPU

To run inference on the CPU instead, pass `--no-nnpa`:

```bash
python /sample/mnist_infer.py --no-nnpa
```

## Known Issues

There are no known open issues with this sample.
