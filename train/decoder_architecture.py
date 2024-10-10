import torch
import torch.nn as nn


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def blockUNet(in_c, out_c, name, transposed=False, bn=True, relu=True, size=4, pad=1, dropout=0.):
    block = nn.Sequential()
    if relu:
        block.add_module('%s_relu' % name, nn.ReLU(inplace=True))
    else:
        block.add_module('%s_leakyrelu' % name, nn.LeakyReLU(0.2, inplace=True))

    if transposed:
        block.add_module('%s_upsam' % name,
                         nn.Upsample(scale_factor=2, mode='bilinear'))  # Upsampling instead of transpose conv
        block.add_module('%s_tconv' % name,
                         nn.Conv2d(in_c, out_c, kernel_size=(size - 1), stride=1, padding=pad, bias=True))

    if bn:
        block.add_module('%s_bn' % name, nn.BatchNorm2d(out_c))

    if dropout > 0.:
        block.add_module('%s_dropout' % name, nn.Dropout2d(dropout, inplace=True))

    return block


# Generator model: Decoder that takes input of size (N, 1, 1)
class Decoder(nn.Module):
    def __init__(self, latent_size, channelExponent=6, dropout=0.):
        super(Decoder, self).__init__()
        channels = int(2 ** channelExponent + 0.5)

        input_size = latent_size + 2
        # Convolutional layer to transform input (N, 1, 1) into (512, 2, 2)
        self.conv_input = nn.Conv2d(input_size, channels * 8, kernel_size=1, stride=1, padding=0)

        # Decoder layers (only upsampling parts, transposed=True)
        self.dlayer6 = blockUNet(channels * 8, channels * 8, 'dlayer6', transposed=True, bn=True, relu=True,
                                 dropout=dropout, size=2, pad=0)
        self.dlayer5 = blockUNet(channels * 8, channels * 8, 'dlayer5', transposed=True, bn=True, relu=True,
                                 dropout=dropout, size=2, pad=0)
        self.dlayer4 = blockUNet(channels * 8, channels * 4, 'dlayer4', transposed=True, bn=True, relu=True,
                                 dropout=dropout)
        self.dlayer3 = blockUNet(channels * 4, channels * 2, 'dlayer3', transposed=True, bn=True, relu=True,
                                 dropout=dropout)
        self.dlayer2b = blockUNet(channels * 2, channels * 2, 'dlayer2b', transposed=True, bn=True, relu=True,
                                  dropout=dropout)
        self.dlayer2 = blockUNet(channels * 2, channels, 'dlayer2', transposed=True, bn=True, relu=True,
                                 dropout=dropout)

        self.dlayer1 = blockUNet(channels, 3, 'dlayer1', transposed=True, bn=False, relu=True)

    def forward(self, x):
        # Apply the first convolutional layer to transform input (N, 1, 1) to (channels * 8, 2, 2)
        x = self.conv_input(x)  # Shape: (batch_size, channels * 8, 1, 1)

        # Pass through decoder layers
        dout6 = self.dlayer6(x)
        dout5 = self.dlayer5(dout6)
        dout4 = self.dlayer4(dout5)
        dout3 = self.dlayer3(dout4)
        dout2b = self.dlayer2b(dout3)
        dout2 = self.dlayer2(dout2b)
        dout1 = self.dlayer1(dout2)

        return dout1
