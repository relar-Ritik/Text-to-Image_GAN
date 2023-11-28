import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from tqdm import trange

class Generator(nn.Module):
    def __init__(self, num_channels=3):
        super(Generator, self).__init__()
        self.ngf = 64  # Number of generator features
        self.num_channels = num_channels

        self.main = nn.Sequential(
            nn.ConvTranspose2d(100, self.ngf * 8, 4, 1, 0, bias=False), # (ngf*8) x 4 x 4
            nn.BatchNorm2d(self.ngf * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(self.ngf * 8, self.ngf * 4, 4, 2, 1, bias=False), # (ngf*4) x 8 x 8
            nn.BatchNorm2d(self.ngf * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(self.ngf * 4, self.ngf * 2, 4, 2, 1, bias=False), # (ngf*2) x 16 x 16
            nn.BatchNorm2d(self.ngf * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(self.ngf * 2, self.ngf, 4, 2, 1, bias=False), # (ngf) x 32 x 32
            nn.BatchNorm2d(self.ngf),
            nn.ReLU(True),
            nn.ConvTranspose2d(self.ngf, self.num_channels, 4, 2, 1, bias=False),  # num_channels x 64 x 64
            nn.Tanh()
        )

    def forward(self, input):
        """The forward function should return batch of images."""

        x = input

        for layer in self.main:
            x = layer(x)
            #print('G: ', x.size())

        return x


class Discriminator(nn.Module):
    def __init__(self, num_channels=3):
        super(Discriminator, self).__init__()
        self.ndf = 64  # Number of discriminator features
        self.num_channels = num_channels

        self.main = nn.Sequential(
            nn.Conv2d(self.num_channels, self.ndf, 4, 2, 1, bias=False),  # (ndf) x 32 x 32
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(self.ndf, self.ndf * 2, 4, 2, 1, bias=False), # (ndf*2) x 16 x 16
            nn.BatchNorm2d(self.ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(self.ndf * 2, self.ndf * 4, 4, 2, 1, bias=False), # (ndf*4) x 8 x 8
            nn.BatchNorm2d(self.ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(self.ndf * 4, self.ndf * 8, 4, 2, 1, bias=False), # (ndf*8) x 4 x 4
            nn.BatchNorm2d(self.ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(self.ndf * 8, 1, 4, 1, 0, bias=False), # 1 x 1 x 1
            nn.Sigmoid()
        )

    def forward(self, input):
        x = input

        for layer in self.main:
            x = layer(x)
            #print('D: ', x.size())

        return x.view(-1, 1).squeeze(1)
