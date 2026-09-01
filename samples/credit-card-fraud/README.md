# Credit Card Fraud Sample

The code sample in this directory uses the
[Credit Card Fraud data set](https://github.com/IBM/TabFormer/tree/main/data/credit_card)
and trains a model. A second script performs inference on the model with the
test data set and displays the results.

The [pytorch README file](../../README.md) contains general information on
downloading and running the samples.

These samples require first downloading the data set from the Internet.

## Prerequisites

The sample depends on packages (`scipy`, `scikit-learn`, `pandas`, `joblib`)
that are not included in the base IBM Z Accelerated for PyTorch container and
require system build tools (`gcc`, `openblas-devel`, etc.) to compile from
source on s390x. Because the base container runs as `ibm-user` (non-root),
`prerequisites.sh` handles this by building a new container image on your
behalf:

1. It passes your chosen PyTorch production image as a build argument to
   `Containerfile`.
2. `Containerfile` temporarily switches to `root` to run the `dnf` and `pip`
   installs, then drops back to `ibm-user` as the runtime user.
3. Once the image is built, `prerequisites.sh` prints the generated image tag
   and the `docker run` command to start the container. The sample scripts are
   mounted read-only at `/scripts` and the named volume is mounted at
   `/workspace` for all output files.

Run the script on the **host** (not from inside a container), passing your
IBM Z Accelerated for PyTorch production image as the argument:

```bash
./prerequisites.sh <base-image>
```

For example:

```bash
./prerequisites.sh icr.io/ibmz/ibmz-accelerated-for-pytorch:1.5.0
```

This builds a local image and prints the generated image tag (e.g.
`pytorch-ccf-sample:20250714-143022`) along with the `docker run` command to
start the container.

## Copying the Data Set into the Container

Once the container is running, open a second terminal on the host and use
`docker cp` to copy the data set into the container:

```bash
# Find the running container ID
docker ps

# Copy the data set
docker cp /path/to/card_transaction.v1.csv <container-id>:/workspace/
```

Then return to the container shell to run the sample.

## Running the Sample

All commands below are run from inside the container, where `/workspace` is
the working directory.

First, train and save the model to disk with the `credit_card_fraud_training.py`
script. Training will take some time.

```bash
python /scripts/credit_card_fraud_training.py
```

This saves the trained model as `ccf_lstm.pt` and the fitted mapper as
`fitted_mapper_v2_lstm.pkl` in `/workspace`. To train a GRU model instead:

```bash
python /scripts/credit_card_fraud_training.py --rnn-type gru
```

You can specify the number of epochs with `--epochs` (default: `20`):

```bash
python /scripts/credit_card_fraud_training.py --epochs 5
```

Once the model has been trained, run the `credit_card_fraud.py` script to run
inference against the model.

```bash
python /scripts/credit_card_fraud.py
```

The script will report the test accuracy. To run inference with the GRU model:

```bash
python /scripts/credit_card_fraud.py --rnn-type gru
```

## Cleanup

When you are finished with the sample, remove the container, image, and workspace
volume:

```bash
docker container prune -f
docker rmi pytorch-ccf-sample:<timestamp>
docker volume rm pytorch-ccf-workspace
```

If you are using rootless Podman, verify no processes are left behind:

```bash
top -u $(whoami)
```

## Known Issues

There are no known open issues with this sample.
