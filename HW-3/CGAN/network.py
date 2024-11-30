import torch.nn as nn
import torch

class Downsample(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride = 2, padding=1, bias=False, padding_mode='reflect'),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.drop = nn.Dropout2d(p=0.5)

    def forward(self, x, with_bn=True, with_drop=True):
        out = self.conv(x)
        if with_bn:
            out = self.bn(out)
        if with_drop:
            out = self.drop(out)
        return out
    
class Upsample(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding = 1, bias=False, padding_mode='zeros'),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.drop = nn.Dropout2d(p=0.5)

    def forward(self, x, with_bn=True, with_drop=True):
        out = self.deconv(x)
        if with_bn:
            out = self.bn(out)
        if with_drop:
            out = self.drop(out)
        return out

class Residual(nn.Module):

    def __init__(self, dim, use_bias):
        super().__init__()
        self.dim = dim
        self.conv = nn.Conv2d(dim, dim, kernel_size = 3, stride = 1, padding = 1, bias=use_bias, padding_mode='reflect')
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.bn = nn.BatchNorm2d(dim)

    def forward(self, x):
        y = self.conv(x)
        y = self.relu(y)
        y = self.bn(y)

        y = self.conv(y)
        y = self.relu(y)
        y = self.bn(y)
    
        return y + x

class Generator(nn.Module):

    def __init__(self, channels=[64, 128, 256, 512]):
        super().__init__()
        self.ch_num = len(channels)
        self.conv1 = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(3, channels[0], kernel_size=7, stride=1, padding=3),
            nn.BatchNorm2d(channels[0]),
            nn.ReLU(inplace=True)
        )
        down_list = []
        up_list = []
        conv_list = []
        for i in range(self.ch_num - 1):
            conv = nn.Sequential(
                Downsample(channels[i], channels[i+1])
            )
            down_list.append(conv)

            k = len(channels) - i - 2
            deconv = nn.Sequential(
                Upsample(channels[k+1], channels[k])
            )
            conv2 = nn.Sequential(
                nn.Conv2d(2*channels[k], channels[k], kernel_size=3, padding=1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.BatchNorm2d(channels[k])
            )
            up_list.append(deconv)
            conv_list.append(conv2)
        
        res_channels = channels[len(channels) - 1]

        self.res = nn.Sequential(
            Residual(res_channels),
            Residual(res_channels)
        )

        self.down_sample = nn.ModuleList(down_list)
        self.up_sample = nn.ModuleList(up_list)
        self.up_conv = nn.ModuleList(conv_list)
        self.last = nn.Sequential(
            nn.ConvTranspose2d(channels[0], 3, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Tanh()
        )

    def forward(self, image):
        x = self.conv1(image)
        x_list = [x]
        for i in range(self.ch_num - 1):
            x_list.append(self.down_sample[i](x_list[-1]))
        
        y = self.res(x_list[-1])
        for i in range(self.ch_num - 1):
            y = self.up_sample[i](y)
            y = torch.cat([y, x_list[-i-2]], dim=1)
            y = self.up_conv[i](y)
        y = self.last(y)
        
        return y
    
class ResGenerator(nn.Module):

    def __init__(self, channels=[64, 128, 256]):
        super().__init__()
        self.ch_num = len(channels)
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, channels[0], kernel_size=7, stride=1, padding=3, bias=False, padding_mode='reflect'),
            nn.BatchNorm2d(channels[0]),
            nn.ReLU(inplace=True)
        )
        down_list = []
        up_list = []
        for i in range(self.ch_num - 1):
            conv = nn.Sequential(
                Downsample(channels[i], channels[i+1])
            )
            down_list.append(conv)

            k = self.ch_num - i - 2
            deconv = nn.Sequential(
                Upsample(channels[k+1], channels[k])
            )
            up_list.append(deconv)
        
        res_channels = channels[self.ch_num - 1]

        self.res = nn.Sequential(
            Residual(res_channels, use_bias=False),
            Residual(res_channels, use_bias=False),
            Residual(res_channels, use_bias=False),
            Residual(res_channels, use_bias=False),
            Residual(res_channels, use_bias=False),
            Residual(res_channels, use_bias=False)
        )

        self.down_sample = nn.ModuleList(down_list)
        self.up_sample = nn.ModuleList(up_list)
        self.last = nn.Sequential(
            nn.Conv2d(channels[0], 3, kernel_size=7, stride=1, padding=3, bias = True, padding_mode='reflect'),
            nn.Tanh()
        )

    def forward(self, image):
        y = self.conv1(image)
        for i in range(self.ch_num - 1):
            y = self.down_sample[i](y)
        y = self.res(y)
        for i in range(self.ch_num - 1):
            y = self.up_sample[i](y)
        y = self.last(y)
        return y
    
class Discriminator(nn.Module):

    def __init__(self, channels=[64, 128, 256, 512]):
        super().__init__()
        self.down1 = nn.Sequential(
                nn.Conv2d(6, channels[0], kernel_size=4, stride=2,padding=1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.InstanceNorm2d(channels[0])
            )
        self.down2 = nn.Sequential(
                nn.Conv2d(channels[0], channels[1], kernel_size=4, stride=2,padding=1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.InstanceNorm2d(channels[1])
            )
        self.down3 = nn.Sequential(
                nn.Conv2d(channels[1], channels[2], kernel_size=4, stride=2,padding=1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.InstanceNorm2d(channels[2])
                
            )
        self.conv = nn.Sequential(
                nn.Conv2d(channels[2], channels[3], kernel_size=4, stride=1,padding=1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.InstanceNorm2d(channels[3])
            )
        self.last = nn.Sequential(
                nn.Conv2d(channels[3], 1, kernel_size=4, stride=1,padding=1)
            )
        
    def forward(self, im_pred, im_source):
        x = torch.cat([im_pred, im_source], dim=1)
        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)
        x = self.conv(x)
        x = self.last(x)

        return x