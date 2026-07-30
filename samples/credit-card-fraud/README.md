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
3. Once the image is built, `prerequisites.sh` creates a `workspace/`
   directory alongside the sample scripts, then starts an interactive shell
   inside the container with:
   - The sample scripts mounted read-only at `/sample`
   - The `workspace/` directory mounted at `/workspace` (writable)

Run the script on the **host** (not from inside a container), passing your
IBM Z Accelerated for PyTorch production image as the argument:

```bash
./prerequisites.sh <base-image> [/path/to/card_transaction.v1.csv]
```

For example:

```bash
./prerequisites.sh icr.io/ibmz/ibmz-accelerated-for-pytorch:1.5.0 /data/card_transaction.v1.csv
```

This builds a local image tagged `ccf-sample:latest` and drops you into an
interactive shell at `/workspace` inside the container. All output files
(model checkpoints, test data, etc.) are written there.

## Obtaining the Data Set

Download the Credit Card Fraud data set. The script will locate
`card_transaction.v1.csv` automatically in the following order:

1. The path passed as the second argument to `prerequisites.sh`
2. The `credit-card-fraud/` sample directory (if you placed it there)
3. The `workspace/` subdirectory (if you placed it there from a previous run)

If the file is not found in any of these locations, the script will print a
warning and you must copy it into `workspace/` manually before running the
sample scripts inside the container.

## Running the Sample

All commands below are run from inside the container, where `/workspace` is
the working directory.

First, train and save the model to disk with the `credit_card_fraud_training.py`
script. Training will take some time.

```bash
python /sample/credit_card_fraud_training.py
```

This saves the trained model as `ccf_lstm.pt` and the fitted mapper as
`fitted_mapper_v2_lstm.pkl` in `/workspace`. To train a GRU model instead:

```bash
python /sample/credit_card_fraud_training.py --rnn-type gru
```

Once the model has been trained, run the `credit_card_fraud.py` script to run
inference against the model.

```bash
python /sample/credit_card_fraud.py
```

The script will report the test accuracy. To run inference with the GRU model:

```bash
python /sample/credit_card_fraud.py --rnn-type gru
```

## Known Issues

There are no known open issues with this sample.
