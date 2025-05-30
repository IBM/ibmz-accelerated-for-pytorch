#!/usr/bin/env python3

# IBM Confidential
# © Copyright IBM Corp. 2025

"""
Credit Card Fraud Inference
"""

import argparse

import numpy as np
import torch

import data_utils


class RNNModel(torch.nn.Module):
    """
    RNN model.
    """

    def __init__(self, rnn_type: str = 'lstm'):
        super().__init__()
        if rnn_type == 'lstm':
            rnn_module = torch.nn.LSTM
        else:
            rnn_module = torch.nn.GRU
        self.rnn = rnn_module(220, 200, num_layers=2, batch_first=True)
        self.fc = torch.nn.Linear(200, 1)
        self.sig = torch.nn.Sigmoid()

    def forward(self, x: torch.Tensor):
        """
        Forward pass.
        """

        # Only use the output, ignoring the state.
        out, _ = self.rnn(x)
        # Only use the final sample of each output sequence.
        out = out[:, -1, :]
        return self.sig(self.fc(out))


def prepare_model(
    rnn_type: str = 'lstm',
    device: str = 'nnpa'
) -> torch.nn.Module:
    """
    Setup to get the model. Return the compiled model.
    """

    pt_model_path = './saved_model/' + rnn_type + '.pt'
    model = torch.load(pt_model_path)
    model.eval()
    model.to(device)

    return model


def main(
    rnn_type: str = 'lstm',
    device: str = 'nnpa',
    batch_size: int = 2000,
    seq_length: int = 7
):
    """
    main
    """

    test_generator = data_utils.prepare_inference_data(batch_size, seq_length)

    model = prepare_model(rnn_type, device)

    print(model)
    for name, param in model.named_parameters():
        print(name, param.size())

    y_pred = []
    y_true = []

    with torch.inference_mode():
        for input_batch, label_batch in test_generator:
            input_batch = torch.as_tensor(
                input_batch, dtype=torch.float32, device=device)
            y_pred.extend(model(input_batch).to('cpu'))
            y_true.extend(label_batch)

    y_pred = torch.stack(y_pred, dim=0)
    y_true = torch.tensor(np.array(y_true))
    correct_prediction = torch.eq(
        torch.round(y_pred).to(torch.int32), y_true)
    accuracy = torch.mean(correct_prediction.to(torch.float32))
    print('Test accuracy:', accuracy.numpy())


if __name__ == '__main__':
    # CLI interface
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--rnn-type',
        type=str.lower,
        choices=['lstm', 'gru'],
        default='lstm',
        help='RNN type used within model (default: lstm)',
    )
    parser.add_argument(
        '--device',
        type=str.lower,
        choices=['nnpa', 'cpu'],
        default='nnpa',
        help='Device used to run model (default: nnpa)',
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=2000,
        help='Batch size for inference data (default: 2000)',
    )
    parser.add_argument(
        '--seq-length',
        type=int,
        default=7,
        help='Sequence length for inference data (default: 7)',
    )
    args = parser.parse_args()

    main(args.rnn_type, args.device, args.batch_size, args.seq_length)
