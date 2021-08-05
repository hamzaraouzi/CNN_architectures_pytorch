import torch
from torch import nn

class C_block(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(C_block, self).__init__()
        self.relu = nn.ReLU()
        self.conv = nn.Conv2d(in_channels, out_channels, **kwargs)
        self.batchnorm = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        return self.relu(self.batchnorm(self.conv(x)))

class I_Auxiliaire(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(I_Auxiliaire, self).__init__()
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.7)
        self.pool = nn.AvgPool2d(kernel_size=5, stride=3)
        self.conv = C_block(in_channels, 128, kernel_size=1)
        self.fc1 = nn.Linear(2048, 1024)
        self.fc2 = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.pool(x)
        x = self.conv(x)
        x = x.reshape(x.shape[0], -1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x

class I_block(nn.Module):
    def __init__(
        self, in_channels, out_1, red_3, out_3, red_5, out_5, out_1pool
    ):
        super(I_block, self).__init__()

        self.branch_1 = C_block(in_channels, out_1, kernel_size=(1,1))

        self.branch_3 = nn.Sequential(
            C_block(in_channels, red_3, kernel_size=(1,1)),
            C_block(red_3, out_3, kernel_size=(3,3), padding=(1,1)),
        )

        self.branch_5 = nn.Sequential(
            C_block(in_channels, red_5, kernel_size=(1,1)),
            C_block(red_5, out_5, kernel_size=(5,5), padding=(2, 2)),
        )

        self.branch1_p = nn.Sequential(
            nn.MaxPool2d(kernel_size=(3,3), stride=(1, 1), padding=(1, 1)),
            C_block(in_channels, out_1pool, kernel_size=(1,1)),
        )

    def forward(self, x):
        return torch.cat(
            [self.branch_1(x), self.branch_3(x), self.branch_5(x), self.branch1_p(x)], 1
        )

class GoogLeNet(nn.Module):
    def __init__(self, in_channels=3,aux_logits=True, num_classes=10):
        super(GoogLeNet, self).__init__()
        assert aux_logits == True or aux_logits == False
        self.aux_logits = aux_logits

        self.conv1 = C_block(
            in_channels=in_channels,
            out_channels=64,
            kernel_size=(7, 7),
            stride=(2, 2),
            padding=(3, 3),
        )

        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv2 = C_block(64, 192, kernel_size=3, stride=1, padding=1)
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # In this order: in_channels, out_1, red_3, out_3, red_5, out_5, out_1pool
        self.inception3a = I_block(192, 64, 96, 128, 16, 32, 32)
        self.inception3b = I_block(256, 128, 128, 192, 32, 96, 64)
        self.maxpool3 = nn.MaxPool2d(kernel_size=(3, 3), stride=2, padding=1)

        self.inception4a = I_block(480, 192, 96, 208, 16, 48, 64)
        self.inception4b = I_block(512, 160, 112, 224, 24, 64, 64)
        self.inception4c = I_block(512, 128, 128, 256, 24, 64, 64)
        self.inception4d = I_block(512, 112, 144, 288, 32, 64, 64)
        self.inception4e = I_block(528, 256, 160, 320, 32, 128, 128)
        self.maxpool4 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.inception5a = I_block(832, 256, 160, 320, 32, 128, 128)
        self.inception5b = I_block(832, 384, 192, 384, 48, 128, 128)

        self.avgpool = nn.AvgPool2d(kernel_size=7, stride=1)
        self.dropout = nn.Dropout(p=0.4)
        self.fc1 = nn.Linear(1024, num_classes)

        if self.aux_logits:
            self.aux1 = I_Auxiliaire(512, num_classes)
            self.aux2 = I_Auxiliaire(528, num_classes)
        else:
            self.aux1 = self.aux2 = None

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.maxpool2(x)

        x = self.inception3a(x)
        x = self.inception3b(x)
        x = self.maxpool3(x)

        x = self.inception4a(x)

        if self.aux_logits and self.training:
            aux1 = self.aux1(x)

        x = self.inception4b(x)
        x = self.inception4c(x)
        x = self.inception4d(x)

        if self.aux_logits and self.training:
            aux2 = self.aux2(x)

        x = self.inception4e(x)
        x = self.maxpool4(x)
        x = self.inception5a(x)
        x = self.inception5b(x)
        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.dropout(x)
        x = self.fc1(x)

        if self.aux_logits and self.training:
            return aux1, aux2, x
        else:
            return x