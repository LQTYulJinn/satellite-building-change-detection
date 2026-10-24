import torch
from torch import nn


class DoubleConv(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )


class Encoder(nn.Module):
    def __init__(self, in_channels, base):
        super().__init__()
        self.blocks = nn.ModuleList([
            DoubleConv(in_channels, base),
            DoubleConv(base, base * 2),
            DoubleConv(base * 2, base * 4),
            DoubleConv(base * 4, base * 8),
        ])
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        features = []
        for block in self.blocks:
            x = block(x)
            features.append(x)
            x = self.pool(x)
        return features, x


class Decoder(nn.Module):
    def __init__(self, base, siamese=False):
        super().__init__()
        factor = 2 if siamese else 1
        self.bottleneck = DoubleConv(base * 8 * factor, base * 16)
        self.up = nn.ModuleList([
            nn.ConvTranspose2d(base * 16, base * 8, 2, stride=2),
            nn.ConvTranspose2d(base * 8, base * 4, 2, stride=2),
            nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2),
            nn.ConvTranspose2d(base * 2, base, 2, stride=2),
        ])
        self.blocks = nn.ModuleList([
            DoubleConv(base * 8 * (1 + factor), base * 8),
            DoubleConv(base * 4 * (1 + factor), base * 4),
            DoubleConv(base * 2 * (1 + factor), base * 2),
            DoubleConv(base * (1 + factor), base),
        ])
        self.head = nn.Conv2d(base, 1, 1)

    def forward(self, skips, x):
        x = self.bottleneck(x)
        for up, block, skip in zip(self.up, self.blocks, reversed(skips)):
            x = block(torch.cat((up(x), skip), dim=1))
        return self.head(x)


class ChangeUNet(nn.Module):
    def __init__(self, base=32, architecture="early"):
        super().__init__()
        if architecture not in ("early", "siamese"):
            raise ValueError("architecture must be early or siamese")
        self.architecture = architecture
        self.encoder = Encoder(6 if architecture == "early" else 3, base)
        self.decoder = Decoder(base, siamese=architecture == "siamese")

    def forward(self, a, b):
        if self.architecture == "early":
            skips, x = self.encoder(torch.cat((a, b), dim=1))
        else:
            skips_a, x_a = self.encoder(a)
            skips_b, x_b = self.encoder(b)
            skips = [torch.cat((fa, fb), dim=1) for fa, fb in zip(skips_a, skips_b)]
            x = torch.cat((x_a, x_b), dim=1)
        return self.decoder(skips, x)
