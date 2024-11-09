import pandas as pd
import torch
from torch import nn

from torch.utils.data import Dataset, DataLoader


class OIBCDataset(Dataset):
    def __init__(self, df, window_size, forecast_size):
        self.df = df
        self.window_size = window_size
        self.forecast_size = forecast_size

        self.to_drop_columns = ['하루전가격(원/kWh)', '확정가격여부', 'ts']

        # self.train_data = self.df.drop(columns=self.to_drop_columns)

        for columns in self.to_drop_columns:
            if 'location' in columns:
                self.to_drop_columns.append(columns)

        self.train_data = self.df
        self.labels = self.df['하루전가격(원/kWh)']

    def __len__(self):
        return len(self.train_data) - self.window_size - self.forecast_size + 1

    def __getitem__(self, idx):
        data = self.train_data.iloc[idx: idx + self.window_size]
        label = self.labels.iloc[idx + self.window_size: idx + self.window_size + self.forecast_size]

        # print(data, label)

        return data, label


def get_dataloader(window_size, forecast_size, batch_size=128):
    df = pd.read_csv('OIBC_2024_DATA/data/train_dataset_with_condition.csv')

    # 임시로 해당 위치만 데이터로 사용
    df = df[df['location_Bonggae-dong'] == True]

    train_size = int(0.9 * len(df))

    train_df = df[:train_size]
    val_df = df[train_size:]

    train_dataset = OIBCDataset(train_df, window_size, forecast_size)
    val_dataset = OIBCDataset(val_df, window_size, forecast_size)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size)

    return train_dataloader, val_dataloader

