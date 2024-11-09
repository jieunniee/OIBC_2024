import torch
from torch import nn


class CNNLSTMRegressor(nn.Module):
    def __init__(self, n_hidden, n_layers, kernel_size=5, feature_size=32):
        super(CNNLSTMRegressor, self).__init__()
        self.n_hidden = n_hidden
        self.n_layers = n_layers

        # 1D Conv 레이어
        self.conv1 = nn.Conv1d(
            in_channels=feature_size,
            out_channels=feature_size * 4,
            kernel_size=kernel_size,
            stride=1
        )
        self.conv2 = nn.Conv1d(
            in_channels=feature_size * 4,
            out_channels=feature_size * 8,
            kernel_size=kernel_size,
            stride=1
        )

        # LSTM 레이어
        self.lstm = nn.LSTM(
            input_size=feature_size * 8,  # Conv2 출력 채널에 맞춤
            hidden_size=n_hidden,
            num_layers=n_layers
            # batch_first=True
        )

        # Fully connected output layer
        self.linear = nn.Linear(in_features=n_hidden, out_features=n_hidden)

    def forward(self, x):
        # print(x)
        # CNN 적용
        x = self.conv1(x.permute(0, 2, 1))  # Reshape for CNN: [batch, channel, sequence]
        x = nn.ReLU()(x)
        x = self.conv2(x)
        x = nn.ReLU()(x)

        # LSTM 입력을 위한 차원 조정
        x = x.permute(0, 2, 1)  # [batch, sequence, features]

        # LSTM 적용
        lstm_out, _ = self.lstm(x)
        # LSTM의 마지막 타임스텝 출력 사용
        last_time_step = lstm_out[:, -1, :]
        # print(lstm_out.shape, last_time_step.shape)
        # y_pred = self.linear(last_time_step)
        # return y_pred
        return last_time_step