import torch
from torch import optim, nn
from tqdm import tqdm

from dataloader import get_dataloader
from dlinear import LTSF_DLinear
from util import CustomLoss


def train():
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    # device = 'cpu'
    WINDOW_SIZE = 288
    FORECAST_SIZE = 28*24
    EPOCH = 30

    print('device:', device)

    model = LTSF_DLinear(
        window_size=WINDOW_SIZE,
        forcast_size=FORECAST_SIZE,  # 28일 * 5분단위
        kernel_size=5,
        onehot_size=12,
        individual=True,
        feature_size=42,  # individual=True일 때만 사용
    )
    train_dataloader, val_dataloader, scaler = get_dataloader(
        window_size=WINDOW_SIZE,
        forecast_size=FORECAST_SIZE,
        batch_size=32
    )
    model.to(device)

    # one = list(train_dataloader)[0]
    # print(one[0].shape, one[1].shape)
    #
    # output = model(one[0])
    # output = output[..., 38]  # 하루전 가격
    # label = one[1]
    # print(output.shape, output)

    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    # criterion = CustomLoss()
    criterion = nn.MSELoss()
    val_criterion = nn.MSELoss()

    for epoch in range(1, EPOCH + 1):
        losses = []
        model.train()
        for train_data, label in (pbar := tqdm(train_dataloader, ncols=75)):
            optimizer.zero_grad()

            train_data = train_data.to(device)
            label = label.to(device).squeeze(2)

            output = model(train_data)

            하루전가격 = output[..., 26]
            # 하루전가격 = output[..., 37]
            # print(하루전가격.shape, label.shape)
            loss = criterion(label, 하루전가격)
            loss.backward()

            losses.append(loss.item())
            pbar.set_description('loss: {:.8f}'.format(sum(losses) / len(losses)))

            optimizer.step()

        val_losses = []
        model.eval()
        for val_data, label in val_dataloader:
            val_data = val_data.to(device)
            label = label.to(device).squeeze(2)

            output = model(val_data)
            하루전가격 = output[..., 26]

            loss = val_criterion(label, 하루전가격)
            val_losses.append(loss)

        print('epoch {}, val loss: {}'.format(epoch, sum(val_losses) / len(val_losses)))


if __name__ == '__main__':
    train()