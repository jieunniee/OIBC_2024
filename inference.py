import pandas as pd
from dlinear import LTSF_DLinear
import torch
from torch import optim, nn
import numpy as np
from time_correct import convert_from_base_time
from dataloader import get_dataloader
from datetime import datetime, timedelta, timezone
import requests
import json
import os

FEATURE_SIZE = 44
WINDOW_SIZE = 7 * 24 * 5
FORECAST_SIZE = 7 * 24 * 5
BATCH_SIZE = 32

df = pd.read_csv('OIBC_2024_DATA/data/train_dataset_with_condition.csv')


def load_checkpoint(model, optimizer, file_folder, epoch):
    """
    저장된 모델의 체크포인트를 로드합니다.

    Parameters:
    - model (torch.nn.Module): 로드할 모델
    - optimizer (torch.optim.Optimizer): 모델의 옵티마이저
    - file_folder (str): 체크포인트가 저장된 폴더 경로
    - epoch (int): 로드할 에포크 번호

    Returns:
    - epoch (int): 로드한 체크포인트의 에포크
    - loss (float): 로드한 체크포인트의 손실 값
    """
    file_path = os.path.join(file_folder, 'checkpoint_{}.pth'.format(epoch))
    if os.path.isfile(file_path):
        checkpoint = torch.load(file_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        print(f"Checkpoint loaded from {file_path}")
        return epoch, loss
    else:
        raise FileNotFoundError(f"No checkpoint found at {file_path}")


model = LTSF_DLinear(
    window_size=WINDOW_SIZE,
    forcast_size=FORECAST_SIZE,  # 28일 * 5분단위
    kernel_size=5,
    onehot_size=12,
    individual=False,
    feature_size=FEATURE_SIZE,  # individual=True일 때만 사용
)
optimizer = optim.AdamW(model.parameters(), lr=5e-5)
load_checkpoint(model, optimizer, 'checkpoint', 3)

model.eval()

train_dataloader, val_dataloader, scaler = get_dataloader(
    window_size=WINDOW_SIZE,
    forecast_size=FORECAST_SIZE,
    batch_size=BATCH_SIZE
)


def inference_per(df, location_name):
    inference_data = df[df[location_name] == True]
    inference_data = inference_data[-WINDOW_SIZE:]
    # print(inference_data.iloc[-1, :]['ts'])
    inference_data = inference_data.drop(columns=['실시간 확정 가격(원/kWh)', '확정가격여부', 'ts'])
    assert WINDOW_SIZE == len(inference_data)

    inference_data = inference_data.values
    inference_data = inference_data.astype(np.float32)
    inference_data = torch.from_numpy(inference_data)
    inference_data = inference_data.unsqueeze(0)

    with torch.no_grad():
        output = model(inference_data)
    output = output.squeeze(0)
    # print(output.shape)
    return output.detach().numpy()


results = np.zeros(840)

locations = []
for column in df.columns:
    if 'location' in column:
        locations.append(column)

for location in locations:
    per = inference_per(df, location)
    per = scaler.inverse_transform(per.reshape(-1, 1)).squeeze(1)
    results += per

# maybe 1730991600 2024-11-08 00:00:00+00:00
last_time = convert_from_base_time(df.iloc[-1, :]['ts'].item())
print(last_time)

results /= len(locations)
print(results.shape)
# results = scaler.inverse_transform(results.reshape(-1, 1)).squeeze(1)
# print(results.shape)

total_results = []
for result in results:
    last_time = last_time + timedelta(minutes=15)
    total_results.append((last_time, result))

predict_start_time = datetime(2024, 11, 11, 0, 15, 0)
predict_end_time = datetime(2024, 11, 12, 0, 0, 0)
to_send = []
for time, value in total_results:
    if (time.day == 11 and (time.hour > 0 or (time.hour == 0 and time.minute != 0))) or (
            time.day == 12 and time.hour == 0):
        if time.minute == 0:
            print(time)
            to_send.append(value.item())

print(len(to_send), to_send)

# API KEY
with open('API_KEY.txt', 'rt') as file:
    API_KEY = file.read()

# JSON 데이터
result = {
    "submit_result": to_send
}
# POST 요청 보내기
response = requests.post('https://research-api.solarkim.com/submissions/cmpt-2024',
                         data=json.dumps(result),
                         headers={
                             'Authorization': f'Bearer {API_KEY}'
                         }).json()

print(response)