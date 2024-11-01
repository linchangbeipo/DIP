import torch.nn as nn

class FullyConvNetwork(nn.Module):

    def __init__(self):
        super().__init__()
         # Encoder (Convolutional Layers)
        self.conv = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
        )
        channels = [8,16,32,64,128]
        n = len(channels)
        for i in range(n-1):
            if channels[i] == channels[i+1]:
                self.conv.add_module(f'conv{i}', nn.Conv2d(channels[i], channels[i+1], kernel_size=3, stride=1, padding=1))
                self.conv.add_module(f'bn{i}', nn.BatchNorm2d(channels[i+1]))
                self.conv.add_module(f'relu{i}', nn.LeakyReLU(0.2, inplace=True))
                self.conv.add_module(f'drop{i}', nn.Dropout2d(p=0.1))
            else:
                self.conv.add_module(f'conv{i}', nn.Conv2d(channels[i], channels[i+1], kernel_size=4, stride=2, padding=1))
                self.conv.add_module(f'bn{i}', nn.BatchNorm2d(channels[i+1]))
                self.conv.add_module(f'relu{i}', nn.LeakyReLU(0.2, inplace=True))
                self.conv.add_module(f'drop{i}', nn.Dropout2d(p=0.1))
                #self.conv.add_module(f'pool{i}', nn.MaxPool2d(kernel_size=2, stride=2))


        
        #Eecoder
        de_channels = [3, 8, 16, 32, 64, 128]
        m = len(de_channels)
        self.deconv = nn.Sequential()
        for i in range(m-1):
            self.deconv.add_module(f'deconv{i}', nn.ConvTranspose2d(de_channels[m-i-1], de_channels[m-i-2], kernel_size=4, stride=2, padding=1))
            self.deconv.add_module(f'debn{i}', nn.BatchNorm2d(de_channels[m-i-2]))
            self.conv.add_module(f'relu{i}', nn.LeakyReLU(0.2, inplace=True))
            self.conv.add_module(f'drop{i}', nn.Dropout2d(p=0.1))
        
        #output layer
        self.output = nn.Tanh()

    def forward(self, x):
        # Encoder forward pass
        features = self.conv(x)
        out = self.deconv(features)
        output = self.output(out)
        return output
    