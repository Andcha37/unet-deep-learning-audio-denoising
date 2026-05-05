import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    """(Convolution -> Batch Normalization -> ReLU) * 2 구조"""
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()

        # 1. 인코더 (Down-sampling)
        self.enc1 = ConvBlock(1, 64)   # 입력: 1채널 (스펙트로그램)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.enc2 = ConvBlock(64, 128)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 2. 바닥층 (Bottleneck)
        self.bottleneck = ConvBlock(128, 256)

        # 3. 디코더 (Up-sampling & Skip Connection)
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(256, 128) # 128(upconv) + 128(skip) = 256

        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(128, 64)  # 64(upconv) + 64(skip) = 128

        # 4. 출력층 (최종 마스크 생성)
        self.final_conv = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x):
        # 인코더
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        
        # 바닥층
        b = self.bottleneck(self.pool2(e2))

        # 디코더 (Skip Connection 포함)
        d2 = self.upconv2(b)
        d2 = torch.cat((d2, e2), dim=1) # e2 정보를 가져와 합침
        d2 = self.dec2(d2)

        d1 = self.upconv1(d2)
        d1 = torch.cat((d1, e1), dim=1) # e1 정보를 가져와 합침
        d1 = self.dec1(d1)

        return self.final_conv(d1)