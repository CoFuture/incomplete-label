# 此处实现u net的子模块
import torch
from torch import nn


class DoubleConv(nn.Module):
    # conv -》 bn -》 relu
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class DownModule(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DownModule, self).__init__()
        self.down = nn.Sequential(
            nn.MaxPool2d(kernel_size=2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.down(x)


class UpModule(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpModule, self).__init__()
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)
        self.conv = DoubleConv(in_channels, out_channels)
        pass

    def forward(self, input1, input2):
        # input1为下采样部分提取的特征 input2为需要上采样的特征
        # b c h w在channel上进行cat 即 1
        output = torch.cat([input1, self.up(input2)], 1)
        output = self.conv(output)
        return output


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1))

    def forward(self, x):
        return self.conv(x)
