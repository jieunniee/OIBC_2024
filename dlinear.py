import torch
from torch import nn


class LTSF_Linear(torch.nn.Module):
    def __init__(self, window_size, forcast_size, individual, feature_size):
        super(LTSF_Linear, self).__init__()
        self.window_size = window_size
        self.forcast_size = forcast_size
        self.individual = individual
        self.channels = feature_size
        if self.individual:
            self.Linear = torch.nn.ModuleList()
            for i in range(self.channels):
                self.Linear.append(torch.nn.Linear(self.window_size, self.forcast_size))
        else:
            self.Linear = torch.nn.Linear(self.window_size, self.forcast_size)

    def forward(self, x):
        if self.individual:
            output = torch.zeros([x.size(0), self.forcast_size, x.size(2)],dtype=x.dtype).to(x.device)
            for i in range(self.channels):
                output[:,:,i] = self.Linear[i](x[:,:,i])
            x = output
        else:
            x = self.Linear(x.permute(0,2,1)).permute(0,2,1)
        return x


class moving_avg(torch.nn.Module):
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = torch.nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class series_decomp(torch.nn.Module):
    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        residual = x - moving_mean
        return moving_mean, residual


class LTSF_DLinear(torch.nn.Module):
    def __init__(self, window_size, forcast_size, kernel_size, individual, feature_size, onehot_size=0):
        super(LTSF_DLinear, self).__init__()
        self.window_size = window_size
        self.forcast_size = forcast_size
        self.onehot_size = onehot_size
        self.decompsition = series_decomp(kernel_size)
        self.decompsition_onehot = series_decomp(1)
        self.fc_onehot = torch.nn.Linear(self.forcast_size + self.onehot_size, self.forcast_size)
        self.individual = individual
        self.channels = feature_size
        if self.individual:
            self.Linear_Seasonal = torch.nn.ModuleList()
            self.Linear_Trend = torch.nn.ModuleList()
            for i in range(self.channels):
                self.Linear_Trend.append(torch.nn.Linear(self.window_size, self.forcast_size))
                self.Linear_Trend[i].weight = torch.nn.Parameter(
                    (1 / self.window_size) * torch.ones([self.forcast_size, self.window_size]))
                self.Linear_Seasonal.append(torch.nn.Linear(self.window_size, self.forcast_size))
                self.Linear_Seasonal[i].weight = torch.nn.Parameter(
                    (1 / self.window_size) * torch.ones([self.forcast_size, self.window_size]))
        else:
            self.Linear_Trend = torch.nn.Linear(self.window_size, self.forcast_size)
            self.Linear_Trend.weight = torch.nn.Parameter(
                (1 / self.window_size) * torch.ones([self.forcast_size, self.window_size]))
            self.Linear_Seasonal = torch.nn.Linear(self.window_size, self.forcast_size)
            self.Linear_Seasonal.weight = torch.nn.Parameter(
                (1 / self.window_size) * torch.ones([self.forcast_size, self.window_size]))

    def forward(self, x):
        # 원핫 인코딩된 데이터는 따로 처리 CNN Kernel = 1
        x_data = x[..., :32]
        x_onehot = x[..., 32:]
        trend_init, seasonal_init = self.decompsition(x_data)
        trend_onehot, seasonal_onehot = self.decompsition_onehot(x_onehot)
        # trend_onehot, seasonal_onehot = x_onehot, x_onehot  # 그대로 넣기
        # print(x_data.shape, trend_init.shape, x_onehot.shape)

        # 원핫 데이터와 concat
        trend_init = torch.concat([trend_init, trend_onehot], dim=2)
        seasonal_init = torch.concat([seasonal_init, seasonal_onehot], dim=2)
        trend_init, seasonal_init = trend_init.permute(0, 2, 1), seasonal_init.permute(0, 2, 1)

        # 이후 원본 DLinear와 같음
        if self.individual:
            trend_output = torch.zeros([trend_init.size(0), trend_init.size(1), self.forcast_size],
                                       dtype=trend_init.dtype).to(trend_init.device)
            seasonal_output = torch.zeros([seasonal_init.size(0), seasonal_init.size(1), self.forcast_size],
                                          dtype=seasonal_init.dtype).to(seasonal_init.device)
            for idx in range(self.channels):
                trend_output[:, idx, :] = self.Linear_Trend[idx](trend_init[:, idx, :])
                seasonal_output[:, idx, :] = self.Linear_Seasonal[idx](seasonal_init[:, idx, :])
        else:
            trend_output = self.Linear_Trend(trend_init)
            seasonal_output = self.Linear_Seasonal(seasonal_init)
        x = seasonal_output + trend_output
        # print(x.shape, x_onehot.shape)
        # x = self.fc_onehot(torch.concat([x, x_onehot], dim=1))
        return x.permute(0, 2, 1)


class DLinearMultiOutput(nn.Module):
    def __init__(self, window_size, forecast_size, feature_size):
        super(DLinearMultiOutput, self).__init__()
        self.linear = nn.Linear(window_size * feature_size, forecast_size)
        self.window_size = window_size
        self.forecast_size = forecast_size
        self.feature_size = feature_size

    def forward(self, x):
        batch_size = x.shape[0]
        x = x.view(batch_size, -1)  # Flatten the input
        out = self.linear(x)
        out = out.view(batch_size, self.forecast_size, self.feature_size)  # Reshape to (batch_size, forecast_size, feature_size)
        return out