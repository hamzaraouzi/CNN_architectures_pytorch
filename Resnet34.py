import torch
import torch.nn as nn
import torch.nn.functional as F


class Residual_blockB(nn.Module):
    def __init__(self, in_channels, out_channels, downsample=False):
        super(Residual_blockB, self).__init__()
        self.downsample = downsample

        stride = 1
        if self.downsample:
            stride = 2
            self.projection = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)
            #TODO i may need BN here


        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),

            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels),

        )

    def forward(self, x):
        # print(x.shape)
        out = self.conv_layers(x)

        # print(out.shape)
        out_2 = self.projection(x) if self.downsample else x.clone()
        # print(out_2.shape)

        return F.relu(out + out_2)


class Resnet34(nn.Module):
    def __init__(self, in_channels=3, num_classes=10):
        super(Resnet34, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes

        self.initial_block = nn.Sequential(
            nn.Conv2d(self.in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.fc = nn.Linear(512, self.num_classes)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))


        self.b64 = nn.ModuleList([])
        self.b64.append(Residual_blockB(in_channels=64, out_channels=64, downsample=False))
        self.b64.append(Residual_blockB(in_channels=64, out_channels=64, downsample=False))
        self.b64.append(Residual_blockB(in_channels=64, out_channels=64, downsample=False))

        self.b128 = nn.ModuleList([])
        self.b128.append(Residual_blockB(in_channels=64, out_channels=128, downsample=True))
        self.b128.append(Residual_blockB(in_channels=128, out_channels=128, downsample=False))
        self.b128.append(Residual_blockB(in_channels=128, out_channels=128, downsample=False))
        self.b128.append(Residual_blockB(in_channels=128, out_channels=128, downsample=False))

        self.b256 = nn.ModuleList([])
        self.b256.append(Residual_blockB(in_channels=128, out_channels=256, downsample=True))
        for _ in range(5):
            self.b256.append(Residual_blockB(in_channels=256, out_channels=256, downsample=False))


        self.b512 = nn.ModuleList([])
        self.b512.append(Residual_blockB(in_channels=256, out_channels=512, downsample=True))
        for _ in range(2):
            self.b512.append(Residual_blockB(in_channels=512, out_channels=512, downsample=False))

    def forward(self, x):
        x = self.initial_block(x)

        for block in self.b64:
            x = block(x)

        for block in self.b128:
            x = block(x)

        for block in self.b256:
            x = block(x)

        for block in self.b512:
            x = block(x)

        x = self.avgpool(x)
        # print(x.shape)
        x = x.reshape(x.shape[0], -1)
        return self.fc(x)


