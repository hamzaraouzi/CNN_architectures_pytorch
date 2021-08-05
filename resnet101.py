
import torch
import torch.nn as nn
import torch.nn.functional as F


class Residual_blockC(nn.Module):
    def __init__(self, in_channels, intermidiate_channels, expand=False, downsample=False):
        super(Residual_blockC, self).__init__()
        self.in_channels = in_channels
        self.downsample = downsample
        self.intermidiate_channels = intermidiate_channels
        self.expand = expand

        stride = 2 if self.downsample else 1

        if self.expand:
            self.projection = nn.Sequential(nn.Conv2d(self.in_channels, self.intermidiate_channels*4, kernel_size=1, stride=stride),
                              nn.BatchNorm2d(self.intermidiate_channels * 4))

        self.conv_layers = nn.Sequential(
            nn.Conv2d(self.in_channels, self.intermidiate_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.intermidiate_channels),
            nn.ReLU(),

            nn.Conv2d(self.intermidiate_channels, self.intermidiate_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(self.intermidiate_channels),
            nn.ReLU(),

            nn.Conv2d(self.intermidiate_channels, self.intermidiate_channels*4, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(self.intermidiate_channels*4)
        )

    def forward(self, x):
        #print(x.shape)
        out = self.conv_layers(x)
        #print(out.shape)

        out_1 = self.projection(x) if self.expand else x.clone()

        #print(out_1.shape, self.downsample)
        return F.relu(out + out_1)


class Resnet101(nn.Module):
    def __init__(self, in_channels=3, num_classes=10):
        super(Resnet101, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes

        self.initial_block = nn.Sequential(
            nn.Conv2d(self.in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.b64 = nn.ModuleList([])
        self.b64.append(Residual_blockC(in_channels=64, intermidiate_channels=64, expand=True, downsample=False))
        self.b64.append(Residual_blockC(in_channels=256, intermidiate_channels=64, expand=False, downsample=False))
        self.b64.append(Residual_blockC(in_channels=256, intermidiate_channels=64, expand=False, downsample=False))


        self.b128 = nn.ModuleList([])
        self.b128.append(Residual_blockC(in_channels=256, intermidiate_channels=128, expand=True, downsample=True))
        for _ in range(3):
            self.b128.append(Residual_blockC(in_channels=128*4, intermidiate_channels=128, expand=False, downsample=False))




        self.b256 = nn.ModuleList([])
        self.b256.append(Residual_blockC(in_channels=512, intermidiate_channels=256, expand=True, downsample=True))
        for _ in range(22):
            self.b256.append(Residual_blockC(in_channels=256*4, intermidiate_channels=256, expand=False, downsample=False))


        self.b512 = nn.ModuleList([])
        self.b512.append(Residual_blockC(in_channels=1024, intermidiate_channels=512, expand=True, downsample=True))
        self.b512.append(Residual_blockC(in_channels=512*4, intermidiate_channels=512, expand=False, downsample=False))
        self.b512.append(Residual_blockC(in_channels=512*4, intermidiate_channels=512, expand=False, downsample=False))

        self.fc = nn.Linear(2048*1*1, self.num_classes)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

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
        x = self.fc(x)

        return x




