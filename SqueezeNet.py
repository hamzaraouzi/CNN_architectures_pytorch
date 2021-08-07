import torch 
import torch.nn as nn

class Fire(nn.module):

    def __init__(
        self,
        in_planes:int,
        squeeze_planes:int,
        expand1x1_planes:int,
        expand3x3_planes:int
        ):
        
        super(Fire,self).__init__()
        
        self.in_planes = in_planes
        self.squeeze_planes = squeeze_planes
        self.expand1x1_planes = expand1x1_planes
        self.expand3x3_planes = expand3x3_planes

        self.squeeze = nn.Conv2d(in_planes,squeeze_planes,kernel_size=1)
        self.squeeze_activation = nn.ReLU()
        
        self.expand1x1= nn.Conv2d(squeeze_planes,expand1x1_planes,kernel_size=1)
        self.expand1x1_activation=nn.ReLU()
        
        self.expand3x3= nn.Conv2d(squeeze_planes,expand3x3_planes,kernel_size=3,padding=1)
        self.expand3x3_activation=nn.ReLU()

    def forward(self,x:torch.Tensor):
        x = self.squeeze_activation(self.squeeze(x))
        return torch.cat([
            self.expand1x1_activation(self.expand1x1(x)),
            self.expand3x3_activation(self.expand3x3(x)),
        ],1)

class SqueezeNet(nn.module):
    def __init__(
        self,
        in_channels = 3,
        num_classes = 10
        ):

        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes

        self.features = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=7, stride=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3,stride=2),
            Fire(96, 16, 64, 64),
            Fire(128, 16, 64, 64),
            Fire(128, 32, 128, 128),
            nn.MaxPool2d(kernel_size=3, stride=2),
            Fire(256, 32, 128, 128),
            Fire(256, 48, 192, 192),
            Fire(384, 48, 192, 192),
            Fire(384, 64, 256, 256),
            nn.MaxPool2d(kernel_size=3, stride=2),
            Fire(512, 64, 256, 256),
        )
        
        last_conv = nn.Conv2d(512, self.num_classes, kernel_size=1)

        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            last_conv,
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1,1))
        )

    def forward(self, x: torch.Tensor):
        
        x = self.features(x)
        x = self.classifier(x)
        
        return torch.flatten(x, 1)