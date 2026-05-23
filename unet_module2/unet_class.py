import torch
import torch.nn as nn


# ================================================================
# Residual ConvBlock
# ================================================================
class ConvBlock(nn.Module):
    """(Conv -> BN -> ReLU) * 2 + Residual Connection + Dropout"""
    def __init__(self, in_channels, out_channels, dropout_p=0.1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout_p)  # 추가: 과적합 방지
        )

        # Residual Connection: 채널 수가 다를 경우 1x1 conv로 맞춰줌
        if in_channels != out_channels:
            self.residual = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.residual = nn.Identity()

    def forward(self, x):
        return self.conv(x) + self.residual(x)  # 추가: Residual Connection


# ================================================================
# 2. CBAM (Channel + Spatial Attention)
# ================================================================
class ChannelAttention(nn.Module):
    """채널별 중요도 학습"""
    def __init__(self, channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, kernel_size=1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        return self.sigmoid(avg_out + max_out)


class SpatialAttention(nn.Module):
    """공간별 중요도 학습"""
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        return self.sigmoid(self.conv(x_cat))


class CBAM(nn.Module):
    """Channel + Spatial Attention 결합"""
    def __init__(self, channels, reduction=16):
        super(CBAM, self).__init__()
        self.channel_att = ChannelAttention(channels, reduction)
        self.spatial_att = SpatialAttention()

    def forward(self, x):
        x = x * self.channel_att(x)
        x = x * self.spatial_att(x)
        return x


# ================================================================
# 4. UNet (전체 구조)
# ================================================================
class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()

        # 인코더 (Down-sampling)
        self.enc1 = ConvBlock(1, 64)
        self.down1 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1)

        self.enc2 = ConvBlock(64, 128)
        self.down2 = nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1)

        self.enc3 = ConvBlock(128, 256)
        self.down3 = nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1)

        self.enc4 = ConvBlock(256, 512)
        self.down4 = nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1)

        # Bottleneck + CBAM
        self.bottleneck = ConvBlock(512, 1024)
        self.cbam = CBAM(1024)  # 추가: Bottleneck에 CBAM 적용

        # 디코더 (Up-sampling)
        self.upconv4 = nn.ConvTranspose2d(1024, 512, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.dec4 = ConvBlock(1024, 512)

        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.dec3 = ConvBlock(512, 256)

        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.dec2 = ConvBlock(256, 128)

        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.dec1 = ConvBlock(128, 64)

        # 출력층
        self.final_conv = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x):
        # 인코더
        e1 = self.enc1(x)
        e2 = self.enc2(self.down1(e1))
        e3 = self.enc3(self.down2(e2))
        e4 = self.enc4(self.down3(e3))

        # Bottleneck + CBAM
        b = self.bottleneck(self.down4(e4))
        b = self.cbam(b)  # 추가

        # 디코더 + Attention Gate + Skip Connection
        d4 = self.upconv4(b)
        d4 = torch.cat((d4, e4), dim=1)
        d4 = self.dec4(d4)

        d3 = self.upconv3(d4)
        d3 = torch.cat((d3, e3), dim=1)
        d3 = self.dec3(d3)

        d2 = self.upconv2(d3)
        d2 = torch.cat((d2, e2), dim=1)
        d2 = self.dec2(d2)

        d1 = self.upconv1(d2)
        d1 = torch.cat((d1, e1), dim=1)
        d1 = self.dec1(d1)

        return torch.sigmoid(self.final_conv(d1))
