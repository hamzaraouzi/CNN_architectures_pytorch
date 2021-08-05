
import torch.nn as nn



class VGG_batchNorm(nn.Module):
    def __init__(self, in_channels=3, num_classes=10):
        super(VGG_batchNorm, self).__init__()

        VGG = [32, 32, 'M', 64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']
        self.in_channels = in_channels
        self.conv_layers = self.create_conv_layers(VGG)

        self.fc_block = nn.Sequential(nn.Linear(512*3*3, 512),
                                      nn.ReLU(),
                                      nn.Linear(512, num_classes))

    def forward(self, x):
        x = self.conv_layers(x)

        #print(x.shape)
        x = x.reshape(x.shape[0], -1)

        x = self.fc_block(x)
        return x

    def create_conv_layers(self, architecture):
        layers = []
        in_channels = self.in_channels

        for x in architecture:
            if type(x) == int:
                out_channels = x
                layers += [
                    nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3, 3), stride=(1, 1),
                              padding=(1, 1)), nn.BatchNorm2d(x), nn.ReLU()]
                in_channels = x

            elif x == 'M':
                layers += [nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))]

        return nn.Sequential(*layers)