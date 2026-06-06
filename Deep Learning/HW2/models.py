import torch
from torch import nn
import torch.nn.functional as F


# AlexNet with smaller params - less layers in every conv layer (decreased channels) and number of neurons in linear layers
class AlexNetSmallerVer(nn.Module):
    def __init__(self, model_params):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 48, kernel_size=11),    # 64x64x3 -> 54x54x96
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 54x54x96 -> 27x27x96
            nn.Conv2d(48, 128, kernel_size=5, padding=2), # 27x27x96 -> 27x27x256
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2), # 27x27x256 -> 13x13x256
            nn.Conv2d(128, 192, kernel_size=3, padding=1),  # 13x13x256 -> 13x13x384
            nn.ReLU(),
            nn.Conv2d(192, 128, kernel_size=3, padding=1),  # 13x13x384 -> 13x13x256
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),  # 13x13x256 -> 6x6x256
            nn.Flatten(), 
            nn.Linear(128 * 6 * 6, 1024),  # FC1
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Dropout(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(1024, 200)
        )

    def forward(self, x):
        x = self.model(x)
        return x

    def predict(self, x):
        x = self.model(x)
        return x


















# this is classic AlexNet, but had to change a few first layers to be able to process 64x64 (not 227x227) images
class AlexNet(nn.Module):
    def __init__(self, model_params):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11),    # 64x64x3 -> 54x54x96
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 54x54x96 -> 27x27x96
            nn.Conv2d(96, 256, kernel_size=5, padding=2), # 27x27x96 -> 27x27x256
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2), # 27x27x256 -> 13x13x256
            nn.Conv2d(256, 384, kernel_size=3, padding=1),  # 13x13x256 -> 13x13x384
            nn.ReLU(),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),  # 13x13x384 -> 13x13x256
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),  # 13x13x256 -> 6x6x256
            nn.Flatten(), 
            nn.Linear(256 * 6 * 6, 4096),  # FC1
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 200)
        )

    def forward(self, x):
        x = self.model(x)
        return x

    def predict(self, x):
        x = self.model(x)
        return x










# https://www.digitalocean.com/community/tutorials/writing-resnet-from-scratch-in-pytorch#
# https://towardsdatascience.com/resnets-why-do-they-perform-better-than-classic-convnets-conceptual-analysis-6a9c82e06e53/
class ResidualBlock(nn.Module):
        def __init__(self, in_channels, out_channels, stride = 1):
            super().__init__()
            self.conv = nn.Sequential(
                            nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride = stride, padding = 1),
                            nn.BatchNorm2d(out_channels),
                            nn.ReLU(), 
                            nn.Conv2d(out_channels, out_channels, kernel_size = 3, stride = 1, padding = 1),
                            nn.BatchNorm2d(out_channels))

            self.relu = nn.ReLU()
            self.in_channels = in_channels
            self.out_channels = out_channels
            self.process_resid = nn.Conv2d(in_channels, out_channels, kernel_size = 1)

        def forward(self, x):
            residual = x
            if self.in_channels != self.out_channels:
                residual = self.process_resid(x)
            out = self.conv(x)
            out += residual
            out = self.relu(out)
            return out



# classic ResNet with little changes in first layers to process 64x64 images
class ResNet(nn.Module):
        def __init__(self, model_params):
            super().__init__()
            self.inplanes = 64
            self.pre_conv = nn.Sequential(
                            nn.Conv2d(3, 64, kernel_size = 5),    # 64x64x3 -> 60x60x64
                            nn.BatchNorm2d(64),
                            nn.ReLU(),
                            nn.Conv2d(64, 64, kernel_size = 5),    # 60x60x64 -> 56x56x64
                            nn.BatchNorm2d(64),
                            nn.ReLU())
            
            self.block1 = nn.Sequential(ResidualBlock(64, 64), ResidualBlock(64, 64))
            self.block2 = nn.Sequential(ResidualBlock(64, 128), ResidualBlock(128, 128), ResidualBlock(128, 128))
            self.block3 = nn.Sequential(ResidualBlock(128, 256), ResidualBlock(256, 256),  ResidualBlock(256, 256))
            self.block4 = nn.Sequential(ResidualBlock(256, 512), ResidualBlock(512, 512))

            self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

            self.linear = nn.Sequential(nn.Flatten(), nn.Linear(7*7*512, 200))


        def forward(self, x):
            x = self.pre_conv(x)      # 64x64x3 -> 56x56x64
            x = self.block1(x)        # 56x56x64 -> 56x56x64
            x = self.maxpool(x)       # 56x56x64 -> 28x28x64
            x = self.block2(x)        # 28x28x64 -> 28x28x128
            x = self.maxpool(x)       # 28x28x128 -> 14x14x128
            x = self.block3(x)        # 14x14x128 -> 14x14x256
            x = self.maxpool(x)       # 14x14x256 -> 7x7x256
            x = self.block4(x)        # 7x7x256 -> 7x7x512
            x = self.linear(x)
            return x













# VGG16 model, but as i had to process 64x64 images, i got 13 conv layers in total )))) so named it 16_to_13
class VGG16_to_13(nn.Module):
    def __init__(self, model_params):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3),                 # 64x64x3 -> 62x62x64
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3),                 # 62x62x64 -> 60x60x64
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3),               # 60x60x64 -> 58x58x64
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3),               # 58x58x64 -> 56x56x64
            nn.ReLU(),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),   # 56x56x128 -> 56x56x256
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),   # 56x56x256 -> 56x56x256
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),   # 56x56x256 -> 56x56x256
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),           # 56x56x256 -> 28x28x256

            nn.Conv2d(64, 128, kernel_size=3, padding=1),   # 28x28x512 -> 28x28x512
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),   # 28x28x512 -> 28x28x512
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),   # 28x28x512 -> 28x28x512
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),           # 28x28x512 -> 14x14x512

            nn.Conv2d(128, 128, kernel_size=3, padding=1),   # 14x14x512 -> 14x14x512
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),   # 14x14x512 -> 14x14x512
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),   # 14x14x512 -> 14x14x512
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),           # 14x14x512 -> 7x7x512

            nn.Flatten(),
            nn.Linear(128 * 7 * 7, 512),     
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 200),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.model(x)
        return x

    def predict(self, x):
        x = self.model(x)
        return x















# here i tried the implementation of the CNN with all layers being DilatedConvs and pseudoresiduals (cut the center of the previous layer and add)
# idea from ChromBPNet model https://github.com/kundajelab/chrombpnet.git
class DilConvNet(nn.Module):
        def __init__(self, model_params):
            super().__init__()
            self.inplanes = 64
            self.pre_conv = nn.Sequential(
                            nn.Conv2d(3, 64, kernel_size = 5),    # 64x64x3 -> 60x60x64
                            nn.BatchNorm2d(64),
                            nn.ReLU())

            self.block1 = nn.Sequential(nn.Conv2d(64, 128, kernel_size = 3, dilation = 2),  nn.BatchNorm2d(128), nn.ReLU())   # 60x60x64 -> 56x56x128
            self.block2 = nn.Sequential(nn.Conv2d(128, 256, kernel_size = 3, dilation = 4),  nn.BatchNorm2d(256), nn.ReLU())  # 56x56x128 -> 48x48x256
            self.block3 = nn.Sequential(nn.Conv2d(256, 512, kernel_size = 3, dilation = 8),  nn.BatchNorm2d(512), nn.ReLU())  # 48x48x256 -> 32x32x512
            self.block4 = nn.Sequential(nn.Conv2d(512, 256, kernel_size = 3, dilation = 8),  nn.BatchNorm2d(256), nn.ReLU())  # 32x32x512 -> 16x16x256
            self.block5 = nn.Sequential(nn.Conv2d(256, 128, kernel_size = 3, dilation = 4),  nn.BatchNorm2d(128), nn.ReLU())  # 16x16x256 -> 8x8x128
            self.block6 = nn.Sequential(nn.Conv2d(128, 64, kernel_size = 3, dilation = 2),  nn.BatchNorm2d(64), nn.ReLU())  # 8x8x128 -> 4x4x64
            self.linear = nn.Sequential(nn.Flatten(), nn.Linear(4*4*64, 200))
            self.rescale1 = nn.Conv2d(64, 128, kernel_size = 1)
            self.rescale2 = nn.Conv2d(128, 256, kernel_size = 1)
            self.rescale3 = nn.Conv2d(256, 512, kernel_size = 1)
            self.rescale4 = nn.Conv2d(512, 256, kernel_size = 1)
            self.rescale5 = nn.Conv2d(256, 128, kernel_size = 1)
            self.rescale6 = nn.Conv2d(128, 64, kernel_size = 1)




        def forward(self, x):
            x = self.pre_conv(x)  

            res = x     # 60x60x64
            x = self.block1(x) # 60x60x64 -> 56x56x128
            x = x + self.rescale1(res[:,:, 2:58, 2:58])

            res = x   # 56x56x128
            x = self.block2(x) # 56x56x128 -> 48x48x256
            x = x + self.rescale2(res[:,:, 4:52, 4:52])

            res = x   # 48x48x256
            x = self.block3(x) # 48x48x256 -> 32x32x512
            x = x + self.rescale3(res[:,:, 8:40, 8:40])

            res = x   # 32x32x512
            x = self.block4(x) # 32x32x512 -> 16x16x256
            x = x + self.rescale4(res[:,:, 8:24, 8:24])

            res = x   # 16x16x256
            x = self.block5(x) # 16x16x256 -> 8x8x128
            x = x + self.rescale5(res[:,:, 4:12, 4:12])

            res = x   # 8x8x128
            x = self.block6(x) # 8x8x128 -> 4x4x64
            x = x + self.rescale6(res[:,:, 2:6, 2:6])

            x = self.linear(x)

            return x












# added batch norm 
class AlexNetSmallerVer_BN(nn.Module):
    def __init__(self, model_params):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 48, kernel_size=11),    # 64x64x3 -> 54x54x96
            nn.BatchNorm2d(48), 
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 54x54x96 -> 27x27x96
            nn.Conv2d(48, 128, kernel_size=5, padding=2), # 27x27x96 -> 27x27x256
            nn.BatchNorm2d(128), 
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2), # 27x27x256 -> 13x13x256
            nn.Conv2d(128, 192, kernel_size=3, padding=1),  # 13x13x256 -> 13x13x384
            nn.BatchNorm2d(192), 
            nn.ReLU(),
            nn.Conv2d(192, 128, kernel_size=3, padding=1),  # 13x13x384 -> 13x13x256
            nn.BatchNorm2d(128), 
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),  # 13x13x256 -> 6x6x256
            nn.Flatten(), 
            nn.Linear(128 * 6 * 6, 1024),  # FC1
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Dropout(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(1024, 200)
        )

    def forward(self, x):
        x = self.model(x)
        return x

    def predict(self, x):
        x = self.model(x)
        return x
















# replaced MaxPool2d to DilConvs
class AlexNetSmallerVer_BN_DilConv(nn.Module):
    def __init__(self, model_params):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 48, kernel_size=11),    # 64x64x3 -> 54x54x96
            nn.BatchNorm2d(48), 
            nn.ReLU(),
            # nn.MaxPool2d(kernel_size=2, stride=2),  # 54x54x96 -> 27x27x96
            nn.Conv2d(48, 48, kernel_size=3, stride = 2, padding = 2, dilation=2),
            nn.Conv2d(48, 128, kernel_size=5, padding=2), # 27x27x96 -> 27x27x256
            nn.BatchNorm2d(128), 
            nn.ReLU(),
            # nn.MaxPool2d(kernel_size=3, stride=2), # 27x27x256 -> 13x13x256
            nn.Conv2d(128, 128, kernel_size=3, stride = 2, padding = 1, dilation=2),
            nn.Conv2d(128, 192, kernel_size=3, padding=1),  # 13x13x256 -> 13x13x384
            nn.BatchNorm2d(192), 
            nn.ReLU(),
            nn.Conv2d(192, 128, kernel_size=3, padding=1),  # 13x13x384 -> 13x13x256
            nn.BatchNorm2d(128), 
            nn.ReLU(),
            # nn.MaxPool2d(kernel_size=3, stride=2),  # 13x13x256 -> 6x6x256
            nn.Conv2d(128, 128, kernel_size=3, stride = 2, padding = 1, dilation=2),
            nn.Flatten(), 
            nn.Linear(128 * 6 * 6, 1024),  # FC1
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Dropout(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(1024, 200)
        )

    def forward(self, x):
        x = self.model(x)
        return x

    def predict(self, x):
        x = self.model(x)
        return x













# and finally added residuals and decreased the number of params
class FINAL_MODEL(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.pre_conv = nn.Sequential( # 64x64x3 -> 64x64x48
            nn.Conv2d(3, 48, kernel_size=5, stride=1, padding=2), 
            nn.BatchNorm2d(48),
            nn.ReLU()
        )

        self.block1 = nn.Sequential(  # 64x64x48 -> 32x32x128
            nn.Conv2d(48, 96, kernel_size=3, stride=2, padding=1, dilation=1),  
            nn.BatchNorm2d(96),
            nn.ReLU(),
            nn.Dropout2d(p=0.2),
            nn.Conv2d(96, 128, kernel_size=3, stride=1, padding=2, dilation=2), 
            nn.BatchNorm2d(128)
        )
        self.res1 = nn.Sequential(
            nn.Conv2d(48, 128, kernel_size=1, stride=2),
            nn.BatchNorm2d(128)
        )

        self.block2 = nn.Sequential( # 32x32x128 -> 16x16x192
            nn.ReLU(),
            nn.Conv2d(128, 192, kernel_size=3, stride=2, padding=1, dilation=1), 
            nn.BatchNorm2d(192),
            nn.ReLU(),
            nn.Dropout2d(p=0.2),
            nn.Conv2d(192, 192, kernel_size=3, stride=1, padding=2, dilation=2),
            nn.BatchNorm2d(192)
        )
        self.res2 = nn.Sequential(
            nn.Conv2d(128, 192, kernel_size=1, stride=2),
            nn.BatchNorm2d(192)
        )

        self.block3 = nn.Sequential( # 32x32x128 -> 8x8x256
            nn.ReLU(),
            nn.Conv2d(192, 256, kernel_size=3, stride=2, padding=1, dilation=1), 
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout2d(p=0.2),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=2, dilation=2),
            nn.BatchNorm2d(256)
        )
        self.res3 = nn.Sequential(
            nn.Conv2d(192, 256, kernel_size=1, stride=2),
            nn.BatchNorm2d(256)
        )

        self.block4 = nn.Sequential( # 8x8x256 -> 8x8x256
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )

        self.linear = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)), # 8x8x256 -> 1x1x256
            nn.Flatten(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 200)
        )

        self.act = nn.ReLU()




    def forward(self, x):
        x = self.pre_conv(x)

        res = self.res1(x)
        x = self.block1(x)
        x = self.act(x + res)

        res = self.res2(x)
        x = self.block2(x)
        x = self.act(x + res)

        res = self.res3(x)
        x = self.block3(x)
        x = self.act(x + res)

        x = self.block4(x)
        x = self.linear(x)
        return x
















