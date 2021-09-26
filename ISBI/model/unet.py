from .unet_part import *
import torch


class UNet(nn.Module):
    def __init__(self, in_channels, out_classes):
        super(UNet, self).__init__()
        # 输入参数：输入图片通道数，输出的结果的通道数
        self.n_channels = in_channels
        self.n_classes = out_classes

        self.input_module = DoubleConv(in_channels, 64)
        self.down1 = DownModule(64, 128)
        self.down2 = DownModule(128, 256)
        self.down3 = DownModule(256, 512)
        self.down4 = DownModule(512, 512)
        self.up1 = UpModule(1024, 512)
        self.up2 = UpModule(768, 256)
        self.up3 = UpModule(384, 128)
        self.up4 = UpModule(192, 64)
        self.output_module = OutConv(64, out_classes)

    def forward(self, x):
        x1 = self.input_module(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        u4 = self.up1(x4, x5)
        u3 = self.up2(x3, u4)
        u2 = self.up3(x2, u3)
        u1 = self.up4(x1, u2)
        out = self.output_module(u1)
        return out


if __name__ == '__main__':
    test_net = UNet(in_channels=3, out_classes=1)
    print(test_net)
