import torch
import torch.nn as nn

class SeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(SeparableConv, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, stride=stride, bias=False, groups=in_channels),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, stride=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.layers(x)


class MobileNetV1(nn.Module):
    def __init__(self, in_channels=3, shallow=False, num_classes=10):
        super(MobileNetV1, self).__init__()
        self.initial_block = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1, stride=2, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )

        self.layers = nn.ModuleList([])
        # 32x112x112
        self.layers.append(SeparableConv(in_channels=32, out_channels=64, stride=1))
        # 64x112x112
        self.layers.append(SeparableConv(in_channels=64, out_channels=128, stride=2))
        # 128x56x56
        self.layers.append(SeparableConv(in_channels=128, out_channels=128, stride=1))
        # 128x56x56
        self.layers.append(SeparableConv(in_channels=128, out_channels=256, stride=2))
        # 256x28x28
        self.layers.append(SeparableConv(in_channels=256, out_channels=256, stride=1))
        # 256x28x28
        self.layers.append(SeparableConv(in_channels=256, out_channels=512, stride=2))
        # 512x14x14
        if not shallow:
            for _ in range(5):
                self.layers.append(SeparableConv(in_channels=512, out_channels=512, stride=1))

        self.layers.append(SeparableConv(in_channels=512, out_channels=1024, stride=2))
        self.layers.append(SeparableConv(in_channels=1024, out_channels=1024,
                                         stride=1))  # in the paper we have stride 2 (probabely a mistake)

        self.avg_pool = nn.AvgPool2d(kernel_size=7, stride=1)
        self.fc = nn.Linear(1024, num_classes)

    def forward(self, x):
        # print(x.shape)
        x = self.initial_block(x)

        for i, layer in enumerate(self.layers):
            # print(i,x.shape)
            x = layer(x)

        # print(x.shape)
        x = self.avg_pool(x)

        x = x.reshape(x.shape[0], -1)
        return self.fc(x)