import numpy as np
import torch
from torch import nn


def calculate_measure(actual, forecast):
    actual = np.array(actual)
    forecast = np.array(forecast)

    positive_index = actual > 0
    negative_index = actual <= 0

    # actual handles values between 0 and -1
    actual[(actual <= 0) & (actual > -1)] = -1
    
    # Number of positive and negative prices
    n1 = np.sum(positive_index) + 1e-7
    n2 = np.sum(negative_index) + 1e-7

    # e1: Positive price prediction error rate
    e1 = (
        np.sum(
            np.abs(actual[positive_index] - forecast[positive_index])
            / np.abs(actual[positive_index])
        )
        / n1
    )

    # e2: Negative price prediction error rate
    e2 = (
        np.sum(
            np.abs(actual[negative_index] - forecast[negative_index])
            / np.abs(actual[negative_index])
        )
        / n2
    )

    TP = np.sum((forecast > 0) & (actual > 0))
    TN = np.sum((forecast <= 0) & (actual <= 0))
    FP = np.sum((forecast > 0) & (actual <= 0))
    FN = np.sum((forecast <= 0) & (actual > 0))

    # Accuracy Calculation
    Accuracy = (TP + TN) / (TP + TN + FP + FN)
    # print(f'Accuracy: {Accuracy}')
    # print(f'e1: {e1}, e2: {e2}')

    e_F = 0.2 * e1 + 0.8 * e2 - (Accuracy - 0.95)

    return e_F


class CustomLoss(nn.Module):
    def __init__(self):
        super(CustomLoss, self).__init__()

    def forward(self, actual, forecast):
        # PyTorch 텐서를 numpy 배열로 변환하여 calculate_measure에 전달
        actual_np = actual.detach().cpu().numpy()
        forecast_np = forecast.detach().cpu().numpy()

        # calculate_measure 함수 호출
        loss_value = calculate_measure(actual_np, forecast_np)

        # 결과를 PyTorch 텐서로 변환하여 반환
        return torch.tensor(loss_value, requires_grad=True)