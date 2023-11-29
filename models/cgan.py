import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from tqdm import trange


class Generator(nn.Module):
    def __init__(self, num_channels=3, embed_size=1024):
        super(Generator, self).__init__()
        self.ngf = 64
        self.noise_dim = 100
        self.num_channels = num_channels

        self.main = nn.Sequential(
            nn.ConvTranspose2d(self.noise_dim + embed_size, self.ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(self.ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(self.ngf * 8, self.ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(self.ngf * 4, self.ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(self.ngf * 2, self.ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(self.ngf, self.num_channels, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (num_channels) x 64 x 64
        )

    def forward(self, z, text_embedding):
        text_embedding = text_embedding.unsqueeze(-1).unsqueeze(-1)
        text_embedding = text_embedding.expand(z.size(0), -1, z.size(2), z.size(3))
        gen_input = torch.cat([z, text_embedding], 1)

        x = gen_input

        for layer in self.main:
            x = layer(x)
            # print('G: ', x.size())

        return x


class Discriminator(nn.Module):
    def __init__(self, embed_size):
        super(Discriminator, self).__init__()
        self.ndf = 64
        self.num_channels = 3

        self.netD_1 = nn.Sequential(
            nn.Conv2d(self.num_channels, self.ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(self.ndf, self.ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(self.ndf * 2, self.ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(self.ndf * 4, self.ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ndf * 8),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # Combine features and text embeddings
        self.netD_2 = nn.Sequential(
            nn.Conv2d(self.ndf * 8 + embed_size, self.ndf * 8, 3, 1, 1, bias=False),
            nn.BatchNorm2d(self.ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(self.ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input, text_embedding):
        x_intermediate = self.netD_1(input)
        text_embedding = text_embedding.unsqueeze(2).unsqueeze(3).expand(-1, -1, x_intermediate.size(2),
                                                                         x_intermediate.size(3))
        d_input = torch.cat([x_intermediate, text_embedding], 1)
        return self.netD_2(d_input).view(-1, 1).squeeze(1), x_intermediate
