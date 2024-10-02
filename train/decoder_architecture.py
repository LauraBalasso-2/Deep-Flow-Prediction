import torch.nn as nn


class Decoder(nn.Module):
    def __init__(self, latent_size):
        super(Decoder, self).__init__()

        n_input = latent_size + 2

        # Define the deconvolution layers
        self.deconv1 = nn.ConvTranspose2d(n_input, 512, kernel_size=4, stride=1, padding=0)  # Output: (512, 4, 4)
        self.bn1 = nn.BatchNorm2d(512)  # BatchNorm after deconv1

        self.deconv2 = nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1)  # Output: (256, 8, 8)
        self.bn2 = nn.BatchNorm2d(256)  # BatchNorm after deconv2

        self.deconv3 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)  # Output: (128, 16, 16)
        self.bn3 = nn.BatchNorm2d(128)  # BatchNorm after deconv3

        self.deconv4 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)  # Output: (64, 32, 32)
        self.bn4 = nn.BatchNorm2d(64)  # BatchNorm after deconv4

        self.deconv5 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)  # Output: (32, 64, 64)
        self.bn5 = nn.BatchNorm2d(32)  # BatchNorm after deconv5

        self.deconv6 = nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1)  # Output: (3, 128, 128)

        self.leaky_relu = nn.LeakyReLU(0.2)

    def forward(self, x):
        x = self.leaky_relu(self.bn1(self.deconv1(x)))
        x = self.leaky_relu(self.bn2(self.deconv2(x)))
        x = self.leaky_relu(self.bn3(self.deconv3(x)))
        x = self.leaky_relu(self.bn4(self.deconv4(x)))
        x = self.leaky_relu(self.bn5(self.deconv5(x)))
        x = self.deconv6(x)  # No activation or batch norm after final layer
        return x

