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
        self.down1 = nn.Conv2d(64, 64, kernel_size=2, stride=2)  # convolution으로 다운샘플링

        self.enc2 = ConvBlock(64, 128)
        self.down2 = nn.Conv2d(128, 128, kernel_size=2, stride=2) # convolution으로 다운샘플링

        self.enc3 = ConvBlock(128, 256)
        self.down3 = nn.Conv2d(256, 256, kernel_size=2, stride=2)

        self.enc4 = ConvBlock(256, 512)
        self.down4 = nn.Conv2d(512, 512, kernel_size=2, stride=2)

        # 2. 바닥층 (Bottleneck)
        self.bottleneck = ConvBlock(512, 1024)

        # 3. 디코더 (Up-sampling & Skip Connection)
        self.upconv4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec4 = ConvBlock(1024, 512)
        
        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = ConvBlock(512, 256)

        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(256, 128) # 128(upconv) + 128(skip) = 256

        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(128, 64)  # 64(upconv) + 64(skip) = 128

        # 4. 출력층 (최종 마스크 생성)
        self.final_conv = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x):
        # 인코더
        e1 = self.enc1(x)
        e2 = self.enc2(self.down1(e1))
        e3 = self.enc3(self.down2(e2))
        e4 = self.enc4(self.down3(e3))
        
        # 바닥층
        b = self.bottleneck(self.down4(e4))

        # 디코더 (Skip Connection 포함)
        d4 = self.upconv4(b)
        d4 = torch.cat((d4, e4), dim=1) # e2 정보를 가져와 합침
        d4 = self.dec4(d4)

        d3 = self.upconv3(d4)
        d3 = torch.cat((d3, e3), dim=1)
        d3 = self.dec3(d3)

        d2 = self.upconv2(d3)
        d2 = torch.cat((d2, e2), dim=1)
        d2 = self.dec2(d2)

        d1 = self.upconv1(d2)
        d1 = torch.cat((d1, e1), dim=1) # e1 정보를 가져와 합침
        d1 = self.dec1(d1)

        return torch.sigmoid(self.final_conv(d1))