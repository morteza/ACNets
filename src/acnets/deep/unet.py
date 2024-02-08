import torch
import torch.nn as nn
import pytorch_lightning as pl


class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_p=0.3):
        super().__init__()

        # TODO set channel sizes

        self.net1 = nn.Sequential(
            nn.BatchNorm2d(1),
            nn.Conv2d(1, 1, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(1),
            nn.Conv2d(1, 1, kernel_size=3, padding=1),
            nn.ReLU()
        )

        self.net2 = nn.Sequential(
            nn.Dropout(dropout_p),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

    def forward(self, x):
        skip = self.net1(x)
        out = self.net2(x)

        return skip, out


class UpBlock(nn.Module):
    def __init__(self, is_final_block=False, n_classes=2):

        # TODO set channel sizes

        super().__init__()
        self.net1 = nn.Sequential(
            nn.BatchNorm2d(1),
            nn.Conv2d(1, 1, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(1),
            nn.Conv2d(1, 1, kernel_size=3, padding=1),
            nn.ReLU()
        )

        if is_final_block:
            self.net2 = nn.Conv2d(1, n_classes, kernel_size=2, stride=2)
        else:
            self.net2 = nn.ConvTranspose2d(1, 1, kernel_size=2, stride=2)

    def forward(self, x, skip):
        x = self.net1(x)
        x = torch.cat([x, skip], dim=1)
        out = self.net2(x)

        return out


class UNet(pl.LightningModule):
    def __init__(self, n_blocks):
        super(UNet, self).__init__()

        self.down = [DownBlock(1, 1) for _ in range(n_blocks)]
        self.center = UpBlock()
        self.up = [UpBlock() for _ in range(n_blocks - 1)]
        self.up.append(UpBlock(is_final_block=True))

        # TODO set channel sizes
        # TODO self.center

    def forward(self, x):

        skips = []
        for down_block in self.down:
            x, skip = down_block(x)
            skips.append(skip)

        x = self.center(x)

        for i, up_block in enumerate(self.up):
            x = up_block(x, skips[i - 1])

        return nn.Sigmoid()(x)
