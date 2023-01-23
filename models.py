import torch
import torch.nn as nn


class EncoderBlock(nn.Module):

    def __init__(self, in_size, out_size, kernel_size=4, padding=1, stride=2, batch_norm=True,
                 leaky_relu_slope=0.2, use_instance_norm=False):
        """
        Convolutional block
        :param in_size: input depth
        :param out_size: output depth
        :param kernel_size: kernel size
        :param padding: padding
        :param stride: stride
        :param batch_norm: whether to use batch normalization/instance normalization
        :param leaky_relu_slope: slope of leaky ReLU
        :param use_instance_norm: use instance normalization if batch size is 1
        """

        super().__init__()

        self.conv_block = nn.Sequential(
            nn.Conv2d(in_size, out_size, kernel_size=kernel_size, padding=padding, stride=stride,
                      bias=False),
            nn.BatchNorm2d(out_size) if (batch_norm and not use_instance_norm) else None,
            nn.InstanceNorm2d(out_size, affine=True) if (
                    batch_norm and use_instance_norm) else None,
            nn.LeakyReLU(inplace=True, negative_slope=leaky_relu_slope))

        # remove None from conv_block
        self.conv_block = nn.Sequential(*[x for x in self.conv_block if x is not None])

        # initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Conv2d):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, (nn.BatchNorm2d, nn.InstanceNorm2d)):
            nn.init.normal_(module.weight.data, 1.0, 0.2)
            nn.init.constant_(module.bias.data, 0)

    def forward(self, x):
        y = self.conv_block(x)

        return y


class DecoderBlock(nn.Module):

    def __init__(self, in_size, out_size, kernel_size=4, padding=1, stride=2, batch_norm=True,
                 apply_dropout=True, dropout_p=0.5, use_instance_norm=False):
        """
        Convolutional block
        :param in_size: input depth
        :param out_size: output depth
        :param kernel_size: kernel size
        :param padding: padding
        :param stride: stride
        :param batch_norm: whether to use batch normalization
        :param apply_dropout: whether to apply dropout
        :param dropout_p: dropout probability
        :param use_instance_norm: use instance normalization if batch size is 1
        """

        super().__init__()

        self.transpose_conv_block = nn.Sequential(
            nn.ConvTranspose2d(in_size, out_size, kernel_size=kernel_size, padding=padding,
                               stride=stride, bias=False),
            nn.BatchNorm2d(out_size) if (batch_norm and not use_instance_norm) else None,
            nn.InstanceNorm2d(out_size, affine=True) if (
                    batch_norm and use_instance_norm) else None,
            nn.Dropout2d(p=dropout_p) if apply_dropout else None,
            nn.ReLU(inplace=True))

        # remove None from conv_block
        self.transpose_conv_block = nn.Sequential(
            *[x for x in self.transpose_conv_block if x is not None])

        # initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.ConvTranspose2d):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, (nn.BatchNorm2d, nn.InstanceNorm2d)):
            nn.init.normal_(module.weight.data, 1.0, 0.2)
            nn.init.constant_(module.bias.data, 0)

    def forward(self, x):
        y = self.transpose_conv_block(x)

        return y


class Generator(nn.Module):

    def __init__(self, in_channels=1, out_channels=3, filters=(
            64, 128, 256, 512, 512, 512, 512, 512), use_instance_norm=False,
                 leaky_relu_slope=0.2):
        """
        Generator network
        :param in_channels: input channels, 1 for grayscale
        :param out_channels: output channels, 3 for RGB
        :param filters: number of filters in each layer
        :param use_instance_norm: use instance normalization if batch size is 1
        :param leaky_relu_slope: slope of leaky ReLU
        """
        super().__init__()

        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        self.last = nn.ConvTranspose2d(2 * filters[0], out_channels, kernel_size=4, padding=1,
                                       stride=2)
        # initialize weights for last layer using xavier uniform
        nn.init.xavier_uniform_(self.last.weight)
        nn.init.constant_(self.last.bias, 0)

        self.final_activation = nn.Tanh()

        # Encoder
        for i, n_filters in enumerate(filters):
            if i == 0:
                self.encoder.append(EncoderBlock(in_channels, n_filters, batch_norm=False,
                                                 use_instance_norm=use_instance_norm,
                                                 leaky_relu_slope=leaky_relu_slope))
            # cannot use instance norm for last layer of encoder because spatial dimensions are 1
            elif i == len(filters) - 1 and use_instance_norm:
                self.encoder.append(EncoderBlock(filters[i - 1], n_filters, batch_norm=False,
                                                 use_instance_norm=use_instance_norm,
                                                 leaky_relu_slope=leaky_relu_slope))
            else:
                self.encoder.append(EncoderBlock(filters[i - 1], n_filters,
                                                 use_instance_norm=use_instance_norm,
                                                 leaky_relu_slope=leaky_relu_slope))

        # Decoder
        for i, n_filters in enumerate(reversed(filters[:-1])):

            if i == 0:
                self.decoder.append(DecoderBlock(filters[-i - 1], n_filters, apply_dropout=False,
                                                 use_instance_norm=use_instance_norm))
            elif 1 <= i <= 2:
                self.decoder.append(
                    DecoderBlock(2 * filters[-i - 1], n_filters, apply_dropout=False,
                                 use_instance_norm=use_instance_norm))
            else:
                self.decoder.append(DecoderBlock(2 * filters[-i - 1], n_filters,
                                                 use_instance_norm=use_instance_norm))

    def forward(self, x):

        # Create a list to store skip-connections
        skip = []

        # Encoder:
        for encoder_step in self.encoder:
            x = encoder_step(x)
            skip.append(x)

        skip = skip[:-1]  # remove last element

        # Decoder:
        for skip_connection, decoder_step in zip(reversed(skip), self.decoder):
            x = decoder_step(x)
            x = torch.cat((x, skip_connection), dim=1)

        # Last layer:
        x = self.last(x)
        x = self.final_activation(x)

        return x


# TODO: make more generic (e.g. for different input sizes)
class Discriminator(nn.Module):
    def __init__(self, leaky_relu_slope=0.2, in_channels=4, filters=(64, 128, 256, 512),
                 use_instance_norm=False):
        """
        Discriminator network
        :param leaky_relu_slope: slope of leaky ReLU
        :param in_channels: input channels, 4 for RGB + grayscale
        :param filters: number of filters in each layer
        :param use_instance_norm: use instance normalization if batch size is 1
        """
        super(Discriminator, self).__init__()

        self.leaky_relu_slope = leaky_relu_slope
        self.in_channels = in_channels
        self.filters = filters

        self.patch_cnn = nn.ModuleList()

        for i, n_filters in enumerate(filters):
            if i == 0:
                self.patch_cnn.append(EncoderBlock(in_channels, n_filters, kernel_size=4,
                                                   batch_norm=False,
                                                   leaky_relu_slope=leaky_relu_slope,
                                                   use_instance_norm=use_instance_norm))
            elif 1 <= i <= 2:
                self.patch_cnn.append(EncoderBlock(filters[i - 1], n_filters, kernel_size=4,
                                                   leaky_relu_slope=leaky_relu_slope,
                                                   use_instance_norm=use_instance_norm))
            else:
                self.patch_cnn.append(EncoderBlock(filters[i - 1], n_filters, kernel_size=4,
                                                   stride=1, leaky_relu_slope=leaky_relu_slope,
                                                   use_instance_norm=use_instance_norm))

        last = nn.Conv2d(filters[-1], 1, kernel_size=4, stride=1, padding=1)
        # initialize weights for last layer using xavier uniform
        nn.init.xavier_uniform_(last.weight)
        nn.init.constant_(last.bias, 0)
        self.patch_cnn.append(last)

    def forward(self, x):
        for patch_cnn_step in self.patch_cnn:
            x = patch_cnn_step(x)
        return x


if __name__ == '__main__':
    # Test the network
    from torchinfo import summary

    x = torch.randn(1, 1, 256, 256, device='cuda')
    y = torch.randn(1, 3, 256, 256, device='cuda')
    use_instance_norm = True if x.shape[0] == 1 else False
    gen = Generator(use_instance_norm=use_instance_norm).to('cuda')
    disc = Discriminator(use_instance_norm=use_instance_norm).to('cuda')
    summary(gen, depth=4, input_data=x)
    summary(disc, depth=4, input_data=torch.cat((x, y), dim=1))
    y = gen(x)
    print(y.shape)
    z = disc(torch.cat([x, y], dim=1))
    print(z.shape)
