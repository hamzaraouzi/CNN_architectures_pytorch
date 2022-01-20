import torch
import torch.nn as nn

class Separable(nn.Module):

  def __init__(self,in_channels,out_1x1,out_channels):
    super(Separable,self).__init__()
    self.conv1x1 = nn.Conv2d(in_channels,out_1x1,kernel_size=1,stride=1)
    
    self.depthwise_conv = nn.Conv2d(out_1x1,out_channels,kernel_size=3,stride=1,padding=1,groups=out_1x1)
    

  def forward(self,x):
    x = self.conv1x1(x)
    x = self.depthwise_conv(x)
    return x


class conv_block(nn.Module):
  def __init__(self, in_channels, out_channels,**kwrags):
    super(conv_block,self).__init__()
    self.relu = nn.ReLU()
    self.conv = nn.Conv2d(in_channels,out_channels,**kwrags)
    self.batchnorm = nn.BatchNorm2d(out_channels)

  def forward(self, x):
    return self.relu(self.batchnorm(self.conv(x)))



class EntryFlow(nn.Module):

  def __init__(self,in_channels=3):
    super(EntryFlow,self).__init__()
    self.conv1 = nn.Sequential(
        conv_block(3,32,kernel_size=3,stride=2),
        conv_block(32,64,kernel_size=3,stride=1,padding=1)
    )


    self.res1 = nn.Sequential(
        nn.Conv2d(64,128,kernel_size=1,stride=2),
        nn.BatchNorm2d(128)
    )

    self.block1 = nn.Sequential(
        Separable(64,32,128),
        nn.ReLU(),
        Separable(128,64,128),
        nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
    )

    self.res2 = nn.Sequential(
        nn.Conv2d(128,256,kernel_size=1,stride=2),
        nn.BatchNorm2d(256)
    )

    self.block2 = nn.Sequential(
        nn.ReLU(),
        Separable(128,64,256),
        nn.ReLU(),
        Separable(256,128,256),
        nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
    )
  
    self.res3 = nn.Sequential(
        nn.Conv2d(256,728,kernel_size=1,stride=2),
        nn.BatchNorm2d(728)
    )

    self.block3 = nn.Sequential(
        nn.ReLU(),
        Separable(256,182,728),
        nn.ReLU(),
        Separable(728,364,728),
        nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
    )

  def forward(self,x):
    x = self.conv1(x)
    x_res = self.res1(x)
    x_block = self.block1(x)
    x = x_res.add(x_block)

    x_res = self.res2(x)
    x_block = self.block2(x)
    x = x_block.add(x_res)

    x_res = self.res3(x)
    x_block = self.block3(x)
    x = x_block.add(x_res)
    return x


class MiddleFlow(nn.Module):
  def __init__(self,in_channels=728):
    super(MiddleFlow,self).__init__()
    self.layers = nn.Sequential(
        nn.ReLU(),
        Separable(in_channels,364,in_channels),
        nn.ReLU(),
        Separable(in_channels,364,in_channels),
        nn.ReLU(),
        Separable(in_channels,364,in_channels),
    )

  def forward(self,x):
    return x.add(self.layers(x))


class ExitFlow(nn.Module):
    def __init__(self,in_channels=3,num_classes=10):
        super(ExitFlow,self).__init__()
        self.res = nn.Sequential(
            nn.Conv2d(728,1024,kernel_size=1,stride=2),
            nn.BatchNorm2d(1024)
        )

        self.block = nn.Sequential(
            nn.ReLU(),
            Separable(728,364,728),
            nn.ReLU(),
            Separable(728,512,1024),
            nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
        )

        self.tail = nn.Sequential(
            Separable(1024,768,1536),
            nn.ReLU(),
            Separable(1536,1024,2048),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(1),
            nn.Linear(2048,1024),
            nn.Dropout(p=0.2),
            nn.ReLU(),
            nn.Linear(1024,512),
            nn.Dropout(p=0.2),
            nn.ReLU(),
            nn.Linear(512,num_classes)
        )

    def forward(self,x):
        x_res =self.res(x)
        x = x_res.add(self.block(x))
        return self.tail(x)

class Xception(nn.Module):
  def mFlows(self):
    l = []
    for i in range(8):
      l.append(MiddleFlow())

    return nn.Sequential(*l)

  
  def __init__(self,in_channels=3,num_classes=10):
    super(Xception,self).__init__()

    self.entry = EntryFlow()
    
    self.middle_flow = self.mFlows()

    self.exit_flow = ExitFlow(728,num_classes)


  def forward(self,x):
    x = self.entry(x)
    x = self.middle_flow(x)
    x = self.exit_flow(x)

    return x


if __name__ == '__main__':
    x = torch.rand((5,3,299,299))
    model = Xception()
    y = model(x)
    print(y.shape)
