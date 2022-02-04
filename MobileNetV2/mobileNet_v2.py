
import torch
import torch.nn as nn
import numpy as np

def conv_bn(in_channels, out_channels, stride):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU6(inplace=True)
    )


def conv_1x1_bn(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels,kernel_size=1, stride=1, padding=0, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU6(inplace=True)
    )

def make_divisible(x, divisible_by=8):
    return int(np.ceil(x * 1./divisible_by) * divisible_by)



class InvertedResidual(nn.Module):
    def __init__(self, in_channels, out_channels, expansion_ratio, stride):
        super(InvertedResidual, self).__init__()
        
        self.use_residual = stride ==1 and in_channels == out_channels

        hidden_dim = int(in_channels * expansion_ratio)

        # I couldn't even overfit one batch without this: I don't know if it's mentioned in the paper
        if expansion_ratio == 1:

            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, out_channels, 1, 1, 0, bias=False),
                nn.BatchNorm2d(out_channels),
            )


        else:
            self.conv = nn.Sequential(
                
                #pointwise convolution
                nn.Conv2d(in_channels, hidden_dim, kernel_size=1, stride= 1, padding=0, bias=False),
                nn.BatchNorm2d(hidden_dim), 
                nn.ReLU6(inplace=True),

                #depthwise convolution
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=stride, padding=1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),

                #pointwise convolution linear
                nn.Conv2d(hidden_dim, out_channels, kernel_size=1, stride= 1, padding=0, bias=False),
                nn.BatchNorm2d(out_channels)

            )
        
    def forward(self, x):
        if self.use_residual:
            return x + self.conv(x)
        
        else:
            return self.conv(x)



class MobileNetV2(nn.Module):
    def __init__(self, in_channels=3, num_classes=10, width_multiplier=1.):
        super(MobileNetV2, self).__init__()

        inverted_residual_parmeters = [
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1]    
        ]

        self.last_channels_dim = make_divisible(1280 * width_multiplier) if width_multiplier > 1.0 else 1280

        self.features = [ conv_bn(in_channels, 32, stride=2)]
        in_channels=32
        
        for t, c, n, s in inverted_residual_parmeters:
            out_channels = make_divisible(c * width_multiplier) if width_multiplier > 1.0 else c

            for i in range(n):
                if i == 0:
                    self.features.append(InvertedResidual(in_channels,out_channels,stride=s, expansion_ratio=t))
                
                else:
                    self.features.append(InvertedResidual(in_channels, out_channels, stride=1, expansion_ratio=t))
                
                in_channels = out_channels

        self.features.append(conv_1x1_bn(in_channels, self.last_channels_dim))    
        self.features = nn.Sequential(*self.features)
        
        self.avg_pool = nn.AvgPool2d(kernel_size=7, stride=1)
        self.classifier = nn.Linear(self.last_channels_dim, num_classes)
        #self.classifier = nn.Conv2d(self.last_channels_dim, num_classes, kernel_size=1)
    

    def forward(self, x):
        x = self.features(x)
        x = self.avg_pool(x)

        x = x.reshape(x.shape[0], -1)
        x = self.classifier(x)
        return x




if __name__ =='__main__':

        model= MobileNetV2(width_multiplier=0.5)
        x = torch.randn(1,3,224, 224)
        y = model(x)
        print(y.shape)
        #b = InvertedResidual(in_channels = 32,out_channels=16, expansion_ratio=1, stride=1)
        #x = torch.randn(1,32,112, 112)
        #y = b(x)
        #print(y.shape)

        #b = InvertedResidual(in_channels = 16,out_channels=24, expansion_ratio=6, stride=2)
        #b2 = InvertedResidual(in_channels = 24,out_channels=24, expansion_ratio=6, stride=1)
        #x = torch.randn(1,16,112, 112)
        #y = b(x)
        #y = b2(y)
        #print(y.shape)
