import torch
import torch.nn as nn
import torch.nn.functional as F

class conv_block(nn.Module):
  def __init__(self, in_channels, out_channels,**kwrags):
    super(conv_block,self).__init__()
    self.relu = nn.ReLU()
    self.conv = nn.Conv2d(in_channels,out_channels,**kwrags)
    self.batchnorm = nn.BatchNorm2d(out_channels)

  def forward(self, x):
    return self.relu(self.batchnorm(self.conv(x)))


class Stem(nn.Module):
    def __init__(self,in_channels=3):
        super(Stem,self).__init__()

        self.conv1 = conv_block(3,32,kernel_size=3,stride=2,padding=0)
        self.conv2 = conv_block(32,32,kernel_size=3,stride=1,padding=0)
        self.conv3 = conv_block(32,64,kernel_size=3,stride=1,padding=1)

        self.maxpool4 = nn.MaxPool2d(kernel_size=3,stride=2)
        self.conv4 = conv_block(64,96,kernel_size=3,stride=2)

        self.branche1 = nn.Sequential(
            conv_block(160,64,kernel_size=1,stride=1),
            conv_block(64,96,kernel_size=3,stride=1,padding=0)
        )

        self.branche2 = nn.Sequential(
            conv_block(160,64,kernel_size=1,stride=1),
            conv_block(64,64,kernel_size=(7,1),stride=1,padding=(3,0)),
            conv_block(64,64,kernel_size=(1,7),stride=1,padding=(0,3)),
            conv_block(64,96,kernel_size=3,stride=1,padding=0)
        )

        self.pool5 = nn.MaxPool2d(kernel_size=3,stride=2,padding=0)
        self.conv5 = conv_block(192,192,kernel_size=3,stride=2,padding=0)
  
    def forward(self,x):  
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x1 = self.maxpool4(x)
        x2 = self.conv4(x)

        x = torch.cat([x1,x2],1)
        x1 = self.branche1(x)
        x2 = self.branche2(x)
        x = torch.cat([x1,x2],1)

        x1= self.pool5(x)
        x2 = self.conv5(x)

        return torch.cat([x1,x2],1)
    

class InceptionA(nn.Module):
  def __init__(self,in_channels):

    super(InceptionA,self).__init__()
    self.branche1 = conv_block(in_channels,32,kernel_size=1,stride=1)

    self.branche2 = nn.Sequential(
        conv_block(in_channels,32,kernel_size=1,stride=1),
        conv_block(32,32,kernel_size=3,stride=1,padding=1)
    )

    self.branche3 = nn.Sequential(
        conv_block(in_channels,32,kernel_size=1,stride=1),
        conv_block(32,48,kernel_size=3,stride=1,padding=1),
        conv_block(48,64,kernel_size=3,stride=1,padding=1)
    )

    self.conv_linear = nn.Conv2d(128,384,kernel_size=1)


  def forward(self,x):
    x  = F.relu(x)
    x1 = self.branche1(x)
    x2 = self.branche2(x)
    x3 = self.branche3(x)

    y = torch.cat([x1,x2,x3],1)
    y = self.conv_linear(y)

    z = x.add(y)

    return z



class InceptionA_red(nn.Module):
    def __init__(self,in_channels):
        super(InceptionA_red,self).__init__()
        self.pool = nn.MaxPool2d(kernel_size=3,stride=2)
        self.conv = conv_block(in_channels,384,kernel_size=3,stride=2,padding=0)
        self.branche = nn.Sequential(
        conv_block(in_channels,256,kernel_size=1,padding=0),
        conv_block(256,256,kernel_size=3,stride=1,padding=1),
        conv_block(256,384,kernel_size=3,stride=2,padding=0)
         )


    def forward(self,x):
        x1 = self.pool(x)
        x2 = self.conv(x)
        x3 = self.branche(x)

        return torch.cat([x1,x2,x3],1)

class InceptionB(nn.Module):
    def __init__(self,in_channels):
        super(InceptionB,self).__init__()

        self.conv1x1 = conv_block(in_channels,192,kernel_size=1,stride=1)
    
        self.branche = nn.Sequential(
        conv_block(in_channels,128,kernel_size=1,stride=1),
        conv_block(128,160,kernel_size=(1,7),stride=1,padding=(0,3)),
        conv_block(160,192,kernel_size=(7,1),stride=1,padding=(3,0))
         )

        self.conv_linear = nn.Conv2d(384,in_channels,kernel_size=1,stride=1)

    def forward(self,x):
        x = F.relu(x)

        x1 = self.conv1x1(x)
        x2 = self.branche(x)

        y = torch.cat([x1,x2],1)
        y = self.conv_linear(y)

        return x.add(y)

class InceptionB_red(nn.Module):
    def __init__(self,in_channels):
        super(InceptionB_red,self).__init__()

        self.maxpool = nn.MaxPool2d(kernel_size=3,stride=2)
        self.branche1= nn.Sequential(
            conv_block(in_channels,256,kernel_size=1,stride=1),
            conv_block(256,384,kernel_size=3,stride=2,padding=0)
        )

        self.branche2= nn.Sequential(
            conv_block(in_channels,256,kernel_size=1,stride=1),
            conv_block(256,288,kernel_size=3,stride=2,padding=0)
        )

        self.branche3 = nn.Sequential(
            conv_block(in_channels,256,kernel_size=1,stride=1),
            conv_block(256,288,kernel_size=3,stride=1,padding=1),
            conv_block(288,320,kernel_size=3,stride=2,padding=0)
        )

    def forward(self,x):
        x1 = self.maxpool(x)
        x2 = self.branche1(x)
        x3 = self.branche2(x)
        x4 = self.branche3(x)

        return torch.cat([x1,x2,x3,x4],1)

class InceptionC(nn.Module):
    def __init__(self,in_channels):
        super(InceptionC,self).__init__()

        self.conv1x1 = conv_block(in_channels,192,kernel_size=1,stride=1)
        self.branche = nn.Sequential(
            conv_block(in_channels,192,kernel_size=1,stride=1),
            conv_block(192,224,kernel_size=(1,3),stride=1,padding=(0,1)),
            conv_block(224,256,kernel_size=(3,1),stride=1,padding=(1,0))
        )

        self.conv_linear = nn.Conv2d(448,in_channels,kernel_size=1,stride=1)

    def forward(self,x):
        x = F.relu(x)

        x1 = self.conv1x1(x)
        x2 = self.branche(x)

        y = torch.cat([x1,x2],1)
        y = self.conv_linear(y)

        return x.add(y)


class Inception_ResnetV2(nn.Module):
    def __init__(self,in_channels=3,num_classes=10):
        super(Inception_ResnetV2,self).__init__()

        self.stem = Stem(in_channels=in_channels)

        l= []
        for i in range(5):
            l.append(InceptionA(384))
    
            self.A_block = nn.Sequential(*l)

            self.A_red = InceptionA_red(384)

        l= []
        for i in range(10):
            l.append(InceptionB(1152))
        
        self.B_block = nn.Sequential(*l)
        self.B_red= InceptionB_red(1152)

        l= []
        for i in range(5):
            l.append(InceptionC(2144))
        
        
        self.C_block = nn.Sequential(*l)
        self.avg_pool = nn.AdaptiveAvgPool2d((1,1))


        self.dropout = nn.Dropout(p=0.2)

        self.fc = nn.Linear(2144,num_classes)
  
    def forward(self,x):
        x = self.stem(x)

        x = self.A_block(x)
        x = F.relu(x)
        x = self.A_red(x)

        x = self.B_block(x)
        x = F.relu(x)
        x = self.B_red(x)
    
        x = self.C_block(x)
        x = self.avg_pool(x)

        x = torch.flatten(x,1)
        x = self.fc(x)

        x = self.dropout(x)

        return x

if __name__ == '__main__':
    x = torch.rand((1,3,299,299))
    model = Inception_ResnetV2(in_channels=3,num_classes=10)
    y= model(x)
    print(y.shape)