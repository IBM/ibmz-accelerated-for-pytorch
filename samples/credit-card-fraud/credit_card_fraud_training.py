#!/usr/bin/env python3

# IBM Confidential
# © Copyright IBM Corp. 2025, 2026

"""
Credit Card Fraud Training
"""

import argparse
import math
import pickle as pk
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import MinMaxScaler
import torch
from torch.utils.data import DataLoader, TensorDataset


SEQ_LENGTH = 7


def time_encoder(x: pd.DataFrame) -> pd.DataFrame:
    """
    Encoder for time data.
    """

    x_hm = x['Time'].str.split(':', expand=True)
    x_date = pd.DataFrame({
        'year': x['Year'],
        'month': x['Month'],
        'day': x['Day'],
        'hour': x_hm[0],
        'minute': x_hm[1]})
    d = pd.to_datetime(x_date).astype(np.int64)
    return pd.DataFrame(d.values, index=x.index, columns=['Year_Month_Day_Time'])


def amt_encoder(x: pd.DataFrame) -> pd.DataFrame:
    """
    Encoder for decimal data.
    """

    return x.map(lambda amt: amt.lstrip('$')).astype(np.float32).map(
        lambda amt: max(1.0, amt)).map(math.log)


def decimal_encoder(x: pd.DataFrame, length: int = 5) -> pd.DataFrame:
    """
    Encoder for integer data.
    """

    col_name = x.columns[0]
    idx = x.index
    x = np.ravel(x)
    x_new = pd.DataFrame(index=idx)
    for i in range(length):
        x_new[f'{col_name}_x{i}'] = np.mod(x, 10)
        x = np.floor_divide(x, 10)
    return x_new.astype(np.int64)


def fraud_encoder(x: pd.DataFrame) -> pd.DataFrame:
    """
    Encoder for boolean data.
    """

    return x.map(lambda v: '1' if v == 'Yes' else '0').astype(int)


def create_test_sample(df: pd.DataFrame, indices: npt.NDArray[np.int64]):
    """
    Writes test data and indices to files.
    """

    rows = indices.shape[0]
    index_array = np.zeros((rows, SEQ_LENGTH), dtype=np.int64)
    for i in range(SEQ_LENGTH):
        index_array[:, i] = indices + 1 - SEQ_LENGTH + i
    uniques = np.unique(index_array.flatten())
    Path('test_10k.csv').unlink(missing_ok=True)
    df.loc[uniques].to_csv('test_10k.csv', index_label='Index')
    Path('test_10k.indices').unlink(missing_ok=True)
    np.savetxt('test_10k.indices', indices.astype(np.int64), fmt='%d')


def create_training_sets(csv_path: Path):
    """
    Reads csv from path and creates training, validation, and test indices.
    """

    x_original = pd.read_csv(csv_path)

    x_original.sort_values(by=['User', 'Card'], inplace=True)
    x_original.reset_index(inplace=True, drop=True)
    x_original.info()

    first = x_original[['User', 'Card']].drop_duplicates()
    f = np.array(first.index)
    print(first)

    drop_list = np.concatenate([np.arange(x, x + SEQ_LENGTH - 1) for x in f])
    index_list = np.setdiff1d(x_original.index.values, drop_list)

    tot_length = index_list.shape[0]
    train_length = tot_length // 2
    validate_length = (tot_length - train_length) * 3 // 5
    test_length = tot_length - train_length - validate_length
    print(tot_length, train_length, validate_length, test_length)

    np.random.seed(1111)
    train_indices = np.random.choice(index_list, train_length, replace=False)
    tv_list = np.setdiff1d(index_list, train_indices)
    validate_indices = np.random.choice(tv_list, validate_length, replace=False)
    test_indices = np.setdiff1d(tv_list, validate_indices)
    print(train_indices, validate_indices, test_indices)

    create_test_sample(x_original, test_indices[:10000])

    return (x_original, train_indices, validate_indices, test_indices)


def gen_training_batch(
        df: pd.DataFrame, mapper: ColumnTransformer,
        indices: npt.NDArray[np.int64], batch_size: int) -> DataLoader:
    """
    Returns a DataLoader with equal split of fraud and non-fraud inputs.
    """

    np.random.seed(98765)
    train_df = df.loc[indices]
    non_fraud_indices = train_df[train_df['Is Fraud?'] == 'No'].index.values
    non_fraud_size = non_fraud_indices.shape[0]
    fraud_indices = train_df[train_df['Is Fraud?'] == 'Yes'].index.values
    fraud_size = fraud_indices.shape[0]

    data_sets = []
    target_sets = []

    num_sets = min(non_fraud_size // fraud_size, 5)

    for i in range(num_sets):
        print('Generating set', i + 1, 'of', num_sets)

        set_indices = np.concatenate(
            (fraud_indices,
             np.random.choice(non_fraud_indices, fraud_size, replace=False)))

        rows = set_indices.shape[0]
        index_array = np.zeros((rows, SEQ_LENGTH), dtype=np.int64)
        for j in range(SEQ_LENGTH):
            index_array[:, j] = set_indices + 1 - SEQ_LENGTH + j

        full_df = df.loc[index_array.flatten()]
        full_df.reset_index(inplace=True, drop=True)
        full_df = mapper.transform(full_df)

        data_buffer = full_df.drop(['Is Fraud?'], axis=1).to_numpy(
            dtype=np.float32).reshape(rows, SEQ_LENGTH, -1)
        target_buffer = full_df['Is Fraud?'].to_numpy(
            dtype=np.float32).reshape(rows, SEQ_LENGTH, 1)

        data_sets.append(torch.from_numpy(data_buffer))
        target_sets.append(torch.from_numpy(target_buffer[:, SEQ_LENGTH - 1, :]))

    data_tensor = torch.concatenate(data_sets, dim=0)
    target_tensor = torch.concatenate(target_sets, dim=0)

    print('training data size:', data_tensor.shape)
    print('training labels size:', target_tensor.shape)

    return DataLoader(TensorDataset(data_tensor, target_tensor),
                      batch_size=batch_size, shuffle=True)


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
        self.fc1 = torch.nn.Linear(200, 1)
        self.act = torch.nn.Sigmoid()

    def forward(self, x: torch.Tensor):
        """
        Forward pass.
        """

        out, _ = self.rnn(x)
        out = out[:, -1, :]
        return self.act(self.fc1(out))


def main(rnn_type: str = 'lstm', batch_size: int = 2048):
    """
    main
    """

    csv_path = Path('./card_transaction.v1.csv')
    x_original, train_indices, _, _ = create_training_sets(csv_path)

    mapper = ColumnTransformer(
        [
            ('Is Fraud?', FunctionTransformer(fraud_encoder), ['Is Fraud?']),
            ('Merchant State', Pipeline([
                ('imputer', SimpleImputer(strategy='constant')),
                ('ordinal', OrdinalEncoder(dtype=np.int64)),
                ('decimal_encoder', FunctionTransformer(decimal_encoder)),
                ('one_hot', OneHotEncoder(sparse_output=False))
            ]), ['Merchant State']),
            ('Zip', Pipeline([
                ('imputer', SimpleImputer(strategy='constant')),
                ('decimal_encoder', FunctionTransformer(decimal_encoder)),
                ('one_hot', OneHotEncoder(sparse_output=False))
            ]), ['Zip']),
            ('Merchant Name', Pipeline([
                ('ordinal', OrdinalEncoder()),
                ('decimal_encoder', FunctionTransformer(decimal_encoder)),
                ('one_hot', OneHotEncoder(sparse_output=False))
            ]), ['Merchant Name']),
            ('Merchant City', Pipeline([
                ('ordinal', OrdinalEncoder()),
                ('decimal_encoder', FunctionTransformer(decimal_encoder)),
                ('one_hot', OneHotEncoder(sparse_output=False))
            ]), ['Merchant City']),
            ('MCC', Pipeline([
                ('ordinal', OrdinalEncoder(dtype=np.int64)),
                ('decimal_encoder', FunctionTransformer(decimal_encoder)),
                ('one_hot', OneHotEncoder(sparse_output=False))
            ]), ['MCC']),
            ('Use Chip', Pipeline([
                ('imputer', SimpleImputer(strategy='constant')),
                ('ordinal', OrdinalEncoder(dtype=np.int64)),
                ('one_hot', OneHotEncoder(sparse_output=False))
            ]), ['Use Chip']),
            ('Errors?', Pipeline([
                ('imputer', SimpleImputer(strategy='constant')),
                ('ordinal', OrdinalEncoder(dtype=np.int64)),
                ('one_hot', OneHotEncoder(sparse_output=False))
            ]), ['Errors?']),
            ('Year_Month_Day_Time', Pipeline([
                ('time_encoder', FunctionTransformer(time_encoder)),
                ('min_max', MinMaxScaler())
            ]), ['Year', 'Month', 'Day', 'Time']),
            ('Amount', Pipeline([
                ('amt_encoder', FunctionTransformer(amt_encoder)),
                ('min_max', MinMaxScaler())
            ]), ['Amount']),
        ],
        verbose_feature_names_out=False
    )
    mapper.set_output(transform='pandas')

    print('Fitting mapper . . .')
    fitted_mapper = mapper.fit(x_original)

    mapper_path = f'fitted_mapper_v2_{rnn_type}.pkl'
    with open(mapper_path, 'wb') as f:
        pk.dump(fitted_mapper, f)

    train_dataloader = gen_training_batch(
        x_original, fitted_mapper, train_indices, batch_size)

    model = RNNModel(rnn_type)
    print(model)

    loss_criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters())

    for epoch in range(20):
        for batch, (data, targets) in enumerate(train_dataloader):
            score = model(data)
            loss = loss_criterion(score, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch % 25 == 0:
                print(f'At epoch: {epoch}, batch: {batch}, loss: {loss}')

    torch.save(model, f'ccf_{rnn_type}.pt')
    print(f'Model saved to ccf_{rnn_type}.pt')


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
        '--batch-size',
        type=int,
        default=2048,
        help='Batch size for training (default: 2048)',
    )
    args = parser.parse_args()

    main(args.rnn_type, args.batch_size)
