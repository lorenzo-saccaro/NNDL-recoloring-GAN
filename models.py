import torch
import torch.nn as nn


# TODO: check generator architecture (more deep?), use strided conv instead of pooling,
#  add Dropout in decoder, replace ReLU with LeakyReLU in encoder

class ConvBlock(nn.Module):

    def __init__(self, in_size, out_size, kernel=3, padding=1):
        """
        Convolutional block
        :param in_size: input depth
        :param out_size: output depth
        :param kernel: kernel size
        :param padding: padding
        """

        super().__init__()

        self.conv_block = nn.Sequential(
            nn.Conv2d(in_size, out_size, kernel_size=kernel, padding=padding, bias=False),
            nn.BatchNorm2d(out_size),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_size, out_size, kernel_size=kernel, padding=padding, bias=False),
            nn.BatchNorm2d(out_size),
            nn.ReLU(inplace=True)
        )

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Conv2d):
            nn.init.xavier_uniform_(module.weight)
        #             if module.bias is not None:
        #                 module.bias.data.zero_()
        elif isinstance(module, nn.BatchNorm2d):
            nn.init.normal_(module.weight.data, 1.0, 0.2)
            nn.init.constant_(module.bias.data, 0)

    def forward(self, x):
        y = self.conv_block(x)

        return y


class Generator(nn.Module):

    def __init__(self, in_channels=1, out_channels=3, channels=[64, 128, 256, 512]):
        """
        Generator network
        :param in_channels: input channels, 1 for grayscale
        :param out_channels: output channels, 3 for RGB
        :param channels: number of channels in each layer of the encoder/decoder
        """
        super().__init__()

        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        self.pool = nn.MaxPool2d((2, 2), stride=2)

        # Encoder
        for c in channels:
            self.encoder.append(ConvBlock(in_channels, c))
            in_channels = c

        # Bottleneck
        self.bottleneck = ConvBlock(channels[-1], 2 * channels[-1])

        # Decoder
        for c in reversed(channels):
            self.decoder.append(nn.ConvTranspose2d(2 * c, c, kernel_size=2, stride=2))
            self.decoder.append(ConvBlock(2 * c, c))

        # Last Layer
        self.last = nn.Conv2d(channels[0], out_channels, kernel_size=1)

    def forward(self, x):

        # Create a list to store skip-connections
        skip = []

        # Encoder:
        for encoder_step in self.encoder:
            x = encoder_step(x)
            # print("enc \t", x.shape)
            skip.insert(0, x)
            x = self.pool(x)

        # Bottleneck:
        x = self.bottleneck(x)
        # print('b_neck \t', x.shape)

        # Decoder:
        # If j is even run the transpose convolution and then the concatenation
        # If j is odd  run the ConvBlock
        for j, decoder_step in enumerate(self.decoder):
            x = decoder_step(x)

            if j % 2 == 0:
                x = torch.cat((x, skip[j // 2]),
                              dim=1)  # Could raise issues for strange shapes of input
                # print("dec \t", x.shape)

        # Last layer:
        x = self.last(x)
        # print("last\t", x.shape)

        return x


# TODO: check discriminator arch, make more generic (e.g. for different input sizes)
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.patch_cnn = nn.Sequential( #Ho usato lo stesso encoder che abbiamo nel Generator
            nn.Conv2d(in_channels=4, out_channels=64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_channels=512, out_channels=1, kernel_size=4, stride=1, padding=1),
        )

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Conv2d):
            nn.init.xavier_uniform_(module.weight)
        #             if module.bias is not None:
        #                 module.bias.data.zero_()
        elif isinstance(module, nn.BatchNorm2d):
            nn.init.normal_(module.weight.data, 1.0, 0.2)
            nn.init.constant_(module.bias.data, 0)

    def forward(self, x):
        x = self.patch_cnn(x)
        return x



if __name__ == '__main__':
    # Test the network
    x = torch.randn(1, 1, 256, 256)
    gen = Generator()
    disc = Discriminator()
    y = gen(x)
    print(y.shape)
    z = disc(torch.cat([y, y], dim=1))
    print(z.shape)
