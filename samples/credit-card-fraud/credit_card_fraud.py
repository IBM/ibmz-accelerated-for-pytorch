#!/usr/bin/env python3

# IBM Confidential
# © Copyright IBM Corp. 2025, 2026

"""
Credit Card Fraud Inference
"""

import argparse
import pickle as pk

import numpy as np
import numpy.typing as npt
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from credit_card_fraud_training import (
    SEQ_LENGTH, time_encoder, amt_encoder, decimal_encoder, fraud_encoder,
    RNNModel)


def gen_test_batch(
        df: pd.DataFrame, mapper,
        indices: npt.NDArray[np.int64], batch_size: int) -> DataLoader:
    """
    Returns a DataLoader for the test indices.
    """

    rows = indices.shape[0]
    index_array = np.zeros((rows, SEQ_LENGTH), dtype=np.int64)
    for i in range(SEQ_LENGTH):
        index_array[:, i] = indices + 1 - SEQ_LENGTH + i

    full_df = df.loc[index_array.flatten()]
    full_df.reset_index(inplace=True, drop=True)
    full_df = mapper.transform(full_df)

    data_buffer = full_df.drop(['Is Fraud?'], axis=1).to_numpy(
        dtype=np.float32).reshape(rows, SEQ_LENGTH, -1)
    target_buffer = full_df['Is Fraud?'].to_numpy(
        dtype=np.float32).reshape(rows, SEQ_LENGTH, 1)

    data_tensor = torch.from_numpy(data_buffer)
    target_tensor = torch.from_numpy(target_buffer[:, SEQ_LENGTH - 1, :])

    return DataLoader(TensorDataset(data_tensor, target_tensor),
                      batch_size=batch_size, shuffle=False)


def main(
    rnn_type: str = 'lstm',
    device: str = 'nnpa',
    batch_size: int = 2048,
):
    """
    main
    """

    x_original = pd.read_csv('test_10k.csv', index_col='Index')
    indices = np.loadtxt('test_10k.indices').astype(np.int64)

    mapper_path = f'fitted_mapper_v2_{rnn_type}.pkl'
    print(f'Loading mapper from {mapper_path} . . .')
    with open(mapper_path, 'rb') as f:
        fitted_mapper = pk.load(f)

    test_dataloader = gen_test_batch(
        x_original, fitted_mapper, indices, batch_size)

    model_path = f'ccf_{rnn_type}.pt'
    print(f'Loading model from {model_path} . . .')
    model = torch.load(model_path, weights_only=False)
    model.eval()
    model.to(device)

    total_correct = 0
    total_samples = 0

    with torch.inference_mode():
        for data, targets in test_dataloader:
            data = data.to(device)
            targets = targets.to(device)
            score = model(data)
            predictions = torch.round(score)
            total_correct += (targets == predictions).sum()
            total_samples += predictions.size(0)

    accuracy = float(total_correct / total_samples) * 100
    print(f'Test accuracy: {accuracy:.2f}%')


if __name__ == '__main__':
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
        default=2048,
        help='Batch size for inference (default: 2048)',
    )
    args = parser.parse_args()

    main(args.rnn_type, args.device, args.batch_size)
