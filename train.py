import torch
from torch import optim, nn
from tqdm import tqdm

from dataloader import get_dataloader
from dlinear import LTSF_DLinear
from util import CustomLoss


def train():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    WINDOW_SIZE = 288
    FORECAST_SIZE = 28*24
    EPOCH = 30

    model = LTSF_DLinear(
        window_size=WINDOW_SIZE,
        forcast_size=FORECAST_SIZE,  # 28일 * 5분단위
        kernel_size=5,
        individual=False,
        feature_size=-1,  # individual=True일 때만 사용
    )
    train_dataloader, val_dataloader = get_dataloader(
        window_size=WINDOW_SIZE,
        forecast_size=FORECAST_SIZE,
        batch_size=1
    )
    model.to(device)

    # one = list(train_dataloader)[0]
    # print(one[0].shape, one[1].shape)
    #
    # output = model(one[0])
    # output = output[..., 38]  # 하루전 가격
    # label = one[1]
    # print(output.shape, output)

    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    # criterion = CustomLoss()
    criterion = nn.MSELoss()

    for epoch in range(1, EPOCH + 1):
        losses = []
        model.train()
        for train_data, label in (pbar := tqdm(train_dataloader, ncols=75)):
            optimizer.zero_grad()

            train_data = train_data.to(device)
            label = label.to(device)

            output = model(train_data)

            하루전가격 = output[..., 38]
            loss = criterion(label, 하루전가격)
            loss.backward()

            losses.append(loss.item())
            pbar.set_description('loss: {:.8f}'.format(sum(losses) / len(losses)))

            optimizer.step()


if __name__ == '__main__':
    train()