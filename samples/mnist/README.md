<!-- markdownlint-disable MD033 -->

# Sample Prerequisites

This sample uses the `python-mnist` package, which can be installed using pip.

```bash
pip install python-mnist
```

# Training on CPU

Training is performed on the CPU.

- You can specify the number of epoch using the `--epochs` arg.
- You can save the model for inference by passing `--save-model`.

```bash
python mnist_training.py --epochs 2 --save-model
```

# Inference on NNPA Device

After running training and saving a model by passing `--save-model` to
`mnist_training.py`, inference can be ran against the saved model.

Inference is performed on the NNPA device by default.

```bash
python mnist_infer.py
```

# Inference on CPU

Inference can be performed on the CPU by passing `--no-nnpa`.

```bash
python mnist_infer.py --no-nnpa
```
