################
#
# Deep Flow Prediction - N. Thuerey, K. Weissenov, H. Mehrotra, N. Mainali, L. Prantl, X. Hu (TUM)
#
# CNN setup and data normalization
#
################

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
    if not transposed:
        block.add_module('%s_conv' % name, nn.Conv2d(in_c, out_c, kernel_size=size, stride=2, padding=pad, bias=True))
    else:
        block.add_module('%s_upsam' % name,
                         nn.Upsample(scale_factor=2, mode='bilinear'))  # Note: old default was nearest neighbor
        # reduce kernel size by one for the upsampling (ie decoder part)
        block.add_module('%s_tconv' % name,
                         nn.Conv2d(in_c, out_c, kernel_size=(size - 1), stride=1, padding=pad, bias=True))
    if bn:
        block.add_module('%s_bn' % name, nn.BatchNorm2d(out_c))
    if dropout > 0.:
        block.add_module('%s_dropout' % name, nn.Dropout2d(dropout, inplace=True))
    return block


class UNet(nn.Module):
    def __init__(self, latent_size, channelExponent=6, dropout=0.):
        super(UNet, self).__init__()
        channels = int(2 ** channelExponent + 0.5)
        self.latent_size = latent_size

        self.layer1 = nn.Sequential()
        self.layer1.add_module('layer1_conv', nn.Conv2d(2, channels, 4, 2, 1, bias=True))

        self.layer2 = blockUNet(channels, channels * 2, 'layer2', transposed=False, bn=True, relu=False,
                                dropout=dropout)
        self.layer2b = blockUNet(channels * 2, channels * 2, 'layer2b', transposed=False, bn=True, relu=False,
                                 dropout=dropout)
        self.layer3 = blockUNet(channels * 2, channels * 4, 'layer3', transposed=False, bn=True, relu=False,
                                dropout=dropout)
        # note the following layer also had a kernel size of 2 in the original version (cf
        # https://arxiv.org/abs/1810.08217) it is now changed to size 4 for encoder/decoder symmetry; to reproduce
        # the old/original results, please change it to 2
        self.layer4 = blockUNet(channels * 4, channels * 8, 'layer4', transposed=False, bn=True, relu=False,
                                dropout=dropout, size=4)  # note, size 4!
        self.layer5 = blockUNet(channels * 8, channels * 8, 'layer5', transposed=False, bn=True, relu=False,
                                dropout=dropout, size=2, pad=0)
        #self.layer6 = blockUNet(channels * 8, channels * 8, 'layer6', transposed=False, bn=False, relu=False,
        #                        dropout=dropout, size=2, pad=0)

        # note, kernel size is internally reduced by one now
        self.dlayer6 = blockUNet(self.latent_size, channels * 8, 'dlayer6', transposed=True, bn=True, relu=True,
                                 dropout=dropout, size=2, pad=0)
        self.dlayer5 = blockUNet(channels * 16, channels * 8, 'dlayer5', transposed=True, bn=True, relu=True,
                                 dropout=dropout, size=2, pad=0)
        self.dlayer4 = blockUNet(channels * 16, channels * 4, 'dlayer4', transposed=True, bn=True, relu=True,
                                 dropout=dropout)
        self.dlayer3 = blockUNet(channels * 8, channels * 2, 'dlayer3', transposed=True, bn=True, relu=True,
                                 dropout=dropout)
        self.dlayer2b = blockUNet(channels * 4, channels * 2, 'dlayer2b', transposed=True, bn=True, relu=True,
                                  dropout=dropout)
        self.dlayer2 = blockUNet(channels * 4, channels, 'dlayer2', transposed=True, bn=True, relu=True,
                                 dropout=dropout)

        self.dlayer1 = blockUNet(channels * 2, 4, 'dlayer1', transposed=True, bn=False, relu=True)

    def forward(self, x, latent_code):
            out1 = self.layer1(x)
            out2 = self.layer2(out1)
            out2b = self.layer2b(out2)
            out3 = self.layer3(out2b)
            out4 = self.layer4(out3)
            out5 = self.layer5(out4)
            out6 = latent_code
            dout6 = self.dlayer6(out6)

            dout6_out5 = torch.cat([dout6, out5], 1)
            dout5 = self.dlayer5(dout6_out5)
            dout5_out4 = torch.cat([dout5, out4], 1)
            dout4 = self.dlayer4(dout5_out4)
            dout4_out3 = torch.cat([dout4, out3], 1)
            dout3 = self.dlayer3(dout4_out3)
            dout3_out2b = torch.cat([dout3, out2b], 1)
            dout2b = self.dlayer2b(dout3_out2b)
            dout2b_out2 = torch.cat([dout2b, out2], 1)
            dout2 = self.dlayer2(dout2b_out2)
            dout2_out1 = torch.cat([dout2, out1], 1)
            dout1 = self.dlayer1(dout2_out1)

            return dout1

