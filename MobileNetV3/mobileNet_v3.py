
import torch
import torch.nn as nn
import numpy as np


def make_divisible(x, divisible_by=8):
    return int(np.ceil(x * 1./divisible_by) * divisible_by)


class HSigmoid(nn.Module):
    def __init__(self):
        super(HSigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=True)
    
    def forward(self, x):
        return self.relu(x + 3)/6

class HSwish(nn.Module):
    def __init__(self):
        super(HSwish, self).__init__()
        self.sigmoid = HSigmoid()
    
    def forward(self, x):
        return x * self.sigmoid(x)



def conv_bn(in_channels, out_channels, stride):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        HSwish()
    )


def conv_1x1_bn(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels,kernel_size=1, stride=1, padding=0, bias=False),
        nn.BatchNorm2d(out_channels),
        HSwish()
    )


class SELayer(nn.Module):
    def __init__(self, channel, reduction=4):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, make_divisible(channel//reduction, 8)),
            nn.ReLU(inplace=True),
            nn.Linear(make_divisible(channel//reduction, 8), channel),
            HSigmoid()
        )
    
    def forward(self, x):
        batch_size, channels, w, h = x.size()
        y = self.avg_pool(x)
        y = y.reshape(batch_size,channels)
        y = self.fc(y)
        y = y.reshape(batch_size, channels, 1, 1)

        return x * y


class InvertedResidual(nn.Module):
    def __init__(self, in_channels, hidden_dim, out_channels,kernel_size, stride, use_se, use_hs):
        super(InvertedResidual, self).__init__()
        
        self.use_residual = stride ==1 and in_channels == out_channels

        
        if in_channels == hidden_dim:

            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size, stride, (kernel_size-1)//2, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                HSwish() if use_hs else nn.ReLU(inplace=True),
                SELayer(hidden_dim) if use_se else nn.Identity(),

                # pw-linear
                nn.Conv2d(hidden_dim, out_channels, 1, 1, 0, bias=False),
                nn.BatchNorm2d(out_channels),
            )


        else:
            self.conv = nn.Sequential(
                
                #pointwise convolution
                nn.Conv2d(in_channels, hidden_dim, kernel_size=1, stride= 1, padding=0, bias=False),
                nn.BatchNorm2d(hidden_dim), 
                HSwish() if use_hs else nn.ReLU(inplace=True),

                #depthwise convolution
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size=kernel_size, stride=stride, padding=kernel_size//2, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                SELayer(hidden_dim) if use_se else nn.Identity(),
                HSwish() if use_hs else nn.ReLU(inplace=True),

                #pointwise convolution linear
                nn.Conv2d(hidden_dim, out_channels, kernel_size=1, stride= 1, padding=0, bias=False),
                nn.BatchNorm2d(out_channels)

            )
        
    def forward(self, x):
        if self.use_residual:
            return x + self.conv(x)
        
        else:
            return self.conv(x)



class MobileNetV3(nn.Module):
    def __init__(self, mode='small', in_channels=3, num_classes=10, width_multiplier=1.):
        super(MobileNetV3, self).__init__()

        
        inverted_residual_parmeters ={ 'large': [
            #in_channles, hidden_dim, out_channels,kernel_size, stride, use_se, use_hs
            [16, 16, 16, 3, 1, False, False],
            [16, 64, 24, 3, 2, False, False],
            [24, 72, 24, 3, 1, False, False],
            [24, 72, 40, 5, 2, True, False],
            [40, 120, 40, 5, 1, True, False],
            [40, 120, 40, 5, 1, True, False],
            [40, 240, 80, 3, 2, False, True],
            [80, 200, 80, 3, 1, False, True],
            [80, 184, 80, 3, 1, False, True],
            [80, 184, 80, 3, 1, False, True],
            [80, 480, 112, 3, 1, True, True],
            [112, 672, 112, 3, 1, True, True],
            [112, 672, 160, 5, 2, True, True],
            [160, 960, 160, 5, 1, True, True],
            [160, 960, 160, 5, 1, True, True]
            ],
            'small':[
                [16, 16, 16, 3, 2, True, False],
                [16, 72, 24, 3, 2, False, False],
                [24, 88, 24, 3, 1, False, False],
                [24, 96, 40, 5, 2, False, False],
                [40, 240, 40, 5, 1, True, True],
                [40, 240, 40, 5, 1, True, True],
                [40, 240, 40, 5, 1, True, True],
                [40, 120, 48, 5, 1, True, True],
                [48, 144, 48, 5, 1, True, True],
                [48, 288, 96, 5, 2, True, True],
                [96, 576, 96, 5, 1, True, True],
                [96, 576, 96, 5, 1, True, True]
            ]
        }

        
        #self.features = 
        
        self.features = [ conv_bn(in_channels, 16, stride=2)]
        for in_channles, hidden_dim, out_channels,kernel_size, stride, use_se, use_hs in inverted_residual_parmeters[mode]:
            #out_channels = make_divisible(c * width_multiplier) if width_multiplier > 1.0 else c
            hidden_dim = make_divisible(hidden_dim * width_multiplier) if width_multiplier > 1.0 else hidden_dim
            self.features.append(InvertedResidual(in_channles, hidden_dim, out_channels,kernel_size, stride, use_se, use_hs))


          
        self.features = nn.Sequential(*self.features)
        


        out1x1 = {'large':960 , 'small': 576}
        out1x1 = make_divisible(out1x1[mode] * width_multiplier) if width_multiplier > 1.0 else out1x1[mode]
        self.conv = conv_1x1_bn(out_channels, out1x1)

        self.avg_pool = nn.AvgPool2d(kernel_size=7, stride=1)

        self.classifier = nn.Sequential(
            nn.Conv2d(out1x1, 1024, kernel_size=1, stride=1),
            HSigmoid(),
            nn.Conv2d(1024, num_classes, kernel_size=1, stride=1)
            )

        #self.classifier = nn.Linear(self.last_channels_dim, num_classes)
        #self.classifier = nn.Conv2d(self.last_channels_dim, num_classes, kernel_size=1)
    

    def forward(self, x):
        
        x = self.features(x)
        x = self.conv(x)
    
        x = self.avg_pool(x)

        x = self.classifier(x)
    
        x = x.reshape(x.shape[0], -1)
    
        return x




#if __name__ =='__main__':

        #model= MobileNetV3(mode='large', width_multiplier=0.5)
        #x = torch.randn(1,3,224, 224)
        #y = model(x)
        #print(y.shape)
