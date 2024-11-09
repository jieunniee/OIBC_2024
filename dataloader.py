import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler
from torch import nn

from torch.utils.data import Dataset, DataLoader


class OIBCDataset(Dataset):
    def __init__(self, train_data, labels, window_size, forecast_size):
        self.window_size = window_size
        self.forecast_size = forecast_size

        self.train_data = train_data
        self.labels = labels

    def __len__(self):
        return len(self.train_data) - self.window_size - self.forecast_size + 1

    def __getitem__(self, idx):
        data = torch.from_numpy(self.train_data[idx: idx + self.window_size])
        label = torch.from_numpy(self.labels[idx + self.window_size: idx + self.window_size + self.forecast_size])

        data = data[..., ]

        return data, label


def get_dataloader(window_size, forecast_size, batch_size=128):
    to_drop_columns = ['실시간 확정 가격(원/kWh)', '확정가격여부', 'ts']
    location_columns = []

    df = pd.read_csv('OIBC_2024_DATA/data/train_dataset_with_condition.csv')

    for idx, column in enumerate(df.columns):
        print(idx, column)

        if 'location' in column:
            location_columns.append(column)


    # 임시로 해당 위치만 데이터로 사용
    # df = df[df['location_Bonggae-dong'] == True]

    # data = df.drop(columns=to_drop_columns).values.astype(np.float32)
    labels = df['실시간 확정 가격(원/kWh)'].values.astype(np.float32)

    data = df.drop(columns=to_drop_columns)
    data_float = data.drop(columns=location_columns).values.astype(np.float32)
    data_onehot = data[location_columns].values.astype(np.float32)
    scaler = MinMaxScaler()
    data_float = scaler.fit_transform(data_float)
    labels = scaler.fit_transform(labels.reshape(-1, 1))

    print(data_float.shape, data_onehot.shape)

    data = np.concat([data_float, data_onehot], axis=1)

    train_size = int(0.9 * len(df))

    train_data = data[:train_size]
    val_data = data[train_size:]

    train_label = labels[:train_size]
    val_label = labels[:train_size]

    train_dataset = OIBCDataset(train_data, train_label, window_size, forecast_size)
    val_dataset = OIBCDataset(val_data, val_label, window_size, forecast_size)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=1)

    return train_dataloader, val_dataloader, scaler

