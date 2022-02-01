import torch
from torch import nn

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ConvBlock, self).__init__()
        
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU()
        
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x

class InceptionF5(nn.Module):
    def __init__(self, in_channels):
        super(InceptionF5, self).__init__()
        
        self.branch1 = nn.Sequential(
            ConvBlock(in_channels, 64, kernel_size=1, stride=1, padding=0),
            ConvBlock(64, 96, kernel_size=3, stride=1, padding=1),
            ConvBlock(96, 96, kernel_size=3, stride=1, padding=1)
        )
        
        self.branch2 = nn.Sequential(
            ConvBlock(in_channels, 48, kernel_size=1, stride=1, padding=0),
            ConvBlock(48, 64, kernel_size=3, stride=1, padding=1)
        )
        
        self.branch3 = nn.Sequential(
            nn.MaxPool2d(3, stride=1, padding=1),
            ConvBlock(in_channels, 64, kernel_size=1, stride=1, padding=0)
        )
        
        self.branch4 = nn.Sequential(
            ConvBlock(in_channels, 64, kernel_size=1, stride=1, padding=0)
        )
        
    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)
        
        return torch.cat([branch1, branch2, branch3, branch4], 1)

class InceptionF6(nn.Module):
    def __init__(self, in_channels, f_7x7):
        super(InceptionF6, self).__init__()
        
        self.branch1 = nn.Sequential(
            ConvBlock(in_channels, f_7x7, kernel_size=1, stride=1, padding=0),
            ConvBlock(f_7x7, f_7x7, kernel_size=(1,7), stride=1, padding=(0,3)),
            ConvBlock(f_7x7, f_7x7, kernel_size=(7,1), stride=1, padding=(3,0)),
            ConvBlock(f_7x7, f_7x7, kernel_size=(1,7), stride=1, padding=(0,3)),
            ConvBlock(f_7x7, 192, kernel_size=(7,1), stride=1, padding=(3,0))
        )
        
        self.branch2 = nn.Sequential(
            ConvBlock(in_channels, f_7x7, kernel_size=1, stride=1, padding=0),
            ConvBlock(f_7x7, f_7x7, kernel_size=(1,7), stride=1, padding=(0,3)),
            ConvBlock(f_7x7, 192, kernel_size=(7,1), stride=1, padding=(3,0))
        )
        
        self.branch3 = nn.Sequential(
            nn.MaxPool2d(3, stride=1, padding=1),
            ConvBlock(in_channels, 192, kernel_size=1, stride=1, padding=0)
        )
        
        self.branch4 = nn.Sequential(
            ConvBlock(in_channels, 192, kernel_size=1, stride=1, padding=0)
        )
        
    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)
        
        return torch.cat([branch1, branch2, branch3, branch4], 1)


class InceptionF7(nn.Module):
    def __init__(self, in_channels):
        super(InceptionF7, self).__init__()
        
        self.branch1 = nn.Sequential(
            ConvBlock(in_channels, 448, kernel_size=1, stride=1, padding=0),
            ConvBlock(448, 384, kernel_size=(3,3), stride=1, padding=1)
        )
        self.branch1_top = ConvBlock(384, 384, kernel_size=(1,3), stride=1, padding=(0,1))
        self.branch1_bot = ConvBlock(384, 384, kernel_size=(3,1), stride=1, padding=(1,0))
        
        
        self.branch2 = ConvBlock(in_channels, 384, kernel_size=1, stride=1, padding=0)
        self.branch2_top = ConvBlock(384, 384, kernel_size=(1,3), stride=1, padding=(0,1))
        self.branch2_bot = ConvBlock(384, 384, kernel_size=(3,1), stride=1, padding=(1,0))
        
        self.branch3 = nn.Sequential(
            nn.MaxPool2d(3, stride=1, padding=1),
            ConvBlock(in_channels, 192, kernel_size=1, stride=1, padding=0)
        )
        
        self.branch4 = nn.Sequential(
            ConvBlock(in_channels, 320, kernel_size=1, stride=1, padding=0)
        )
        
    def forward(self, x):
        branch1 = self.branch1(x)
        branch1 = torch.cat([self.branch1_top(branch1), self.branch1_bot(branch1)], 1)
        
        branch2 = self.branch2(x)
        branch2 = torch.cat([self.branch2_top(branch2), self.branch2_bot(branch2)], 1)
        
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)
        
        return torch.cat([branch1, branch2, branch3, branch4], 1)

class InceptionRed(nn.Module):
    def __init__(self, in_channels, f_3x3_r, add_ch=0):
        super(InceptionRed, self).__init__()
        
        self.branch1 = nn.Sequential(
            ConvBlock(in_channels, f_3x3_r, kernel_size=1, stride=1, padding=0),
            ConvBlock(f_3x3_r, 178 + add_ch, kernel_size=3, stride=1, padding=1),
            ConvBlock(178 + add_ch, 178 + add_ch, kernel_size=3, stride=2, padding=0)
        )
        
        self.branch2 = nn.Sequential(
            ConvBlock(in_channels, f_3x3_r, kernel_size=1, stride=1, padding=0),
            ConvBlock(f_3x3_r, 302 + add_ch, kernel_size=3, stride=2, padding=0)
        )
        
        self.branch3 = nn.Sequential(
            nn.MaxPool2d(3, stride=2, padding=0)
        )
        
    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        
        return torch.cat([branch1, branch2, branch3], 1)

class InceptionAux(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(InceptionAux, self).__init__()
        
        self.pool = nn.AdaptiveAvgPool2d((4,4))
        self.conv = nn.Conv2d(in_channels, 128, kernel_size=1, stride=1, padding=0)
        self.act = nn.ReLU()
        self.fc1 = nn.Linear(2048, 1024)
        self.dropout = nn.Dropout(0.7)
        self.fc2 = nn.Linear(1024, num_classes)
    
    def forward(self, x):
        x = self.pool(x)
        
        x = self.conv(x)
        x = self.act(x)
    
        x = torch.flatten(x, 1)
        
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        
        x = self.fc2(x)
        
        return x


class InceptionV2(nn.Module):
    
    def __init__(self, num_channels = 3 , num_classes = 10, aux = True):
        super(InceptionV2, self).__init__()

        self.aux = aux
        
        self.conv1 = ConvBlock(3, 32, kernel_size=3, stride=2, padding=0)
        self.conv2 = ConvBlock(32, 32, kernel_size=3, stride=1, padding=0)
        self.conv3 = ConvBlock(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(3, stride=2, padding=0)
        self.conv4 = ConvBlock(64, 80, kernel_size=3, stride=1, padding=0)
        self.conv5 = ConvBlock(80, 192, kernel_size=3, stride=2, padding=0)
        self.conv6 = ConvBlock(192, 288, kernel_size=3, stride=1, padding=1)
        
        self.inception3a = InceptionF5(288)
        self.inception3b = InceptionF5(288)
        self.inception3c = InceptionF5(288)
        
        self.inceptionRed1 = InceptionRed(288,f_3x3_r=64, add_ch=0)
        
        self.inception4a = InceptionF6(768, f_7x7=128)
        self.inception4b = InceptionF6(768, f_7x7=160)
        self.inception4c = InceptionF6(768, f_7x7=160)
        self.inception4d = InceptionF6(768, f_7x7=160)
        self.inception4e = InceptionF6(768, f_7x7=192)
        
        self.inceptionRed2 = InceptionRed(768,f_3x3_r=192, add_ch=16)
        
        self.aux = InceptionAux(768, num_classes) 
        
        self.inception5a = InceptionF7(1280)
        self.inception5b = InceptionF7(2048)
        
        self.pool6 = nn.AdaptiveAvgPool2d((1,1))
        self.dropout = nn.Dropout(0.4)
        self.fc = nn.Linear(2048, num_classes)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.pool1(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        
        x = self.inception3a(x)
        x = self.inception3b(x)
        x = self.inception3c(x)

        x = self.inceptionRed1(x)
        
        x = self.inception4a(x)
        x = self.inception4b(x)
        x = self.inception4c(x)
        x = self.inception4d(x)
        x = self.inception4e(x)
        
        if self.aux:
            aux = self.aux(x)
        
        x = self.inceptionRed2(x)    
        x = self.inception5a(x)
        x = self.inception5b(x)
        
        x = self.pool6(x)
        x = self.dropout(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        if self.aux:
            return x, aux
        return x, aux








