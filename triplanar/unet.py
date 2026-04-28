import torch
import torch.nn as nn


class UNet(nn.Module):
    def __init__(self, channels, classes):
        super(UNet, self).__init__()
        self.start = ConvLayer(channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.bridge = ConvLayer(512, 512)
        self.up1 = Up(512, 256)
        self.up2 = Up(256, 128)
        self.up3 = Up(128, 64)
        self.end = nn.Conv2d(64, classes, kernel_size=1)

    def forward(self, x):
        x1 = self.start(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x4 = self.bridge(x4)
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        logits = self.end(x)
        return logits


class ConvLayer(nn.Module):
    def __init__(self, inc, outc):
        super().__init__()
        self.conv_layer = nn.Sequential(
            nn.Conv2d(inc, outc, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(outc),
            nn.ReLU(inplace=True),
            nn.Conv2d(outc, outc, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(outc),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        """Conv2d > InstanceNorm2d > ReLU > Conv2d > InstanceNorm2d > ReLU"""
        return self.conv_layer(x)


class Down(nn.Module):
    def __init__(self, inc, outc):
        super().__init__()
        self.down_layer = nn.Sequential(nn.MaxPool2d(2), ConvLayer(inc, outc))

    def forward(self, x):
        """MaxPool2d > ConvLayer"""
        return self.down_layer(x)


class Up(nn.Module):
    def __init__(self, inc, outc):
        super().__init__()
        self.up = nn.ConvTranspose2d(inc, inc // 2, kernel_size=2, stride=2)
        self.conv = ConvLayer(inc, outc)

    def forward(self, x1, x2):
        """ConvTranspose2d > ConvLayer"""
        x1 = self.up(x1)
        assert x1.shape[2:] == x2.shape[2:]
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)
