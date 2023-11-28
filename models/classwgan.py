import torch
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, image_size=64, inp_channels=3, noise_dim=100, embed_dims=1024):
        super().__init__()
        self.image_size = image_size
        self.inp_channels = inp_channels
        self.noise_dim = noise_dim
        self.embed_dims = embed_dims

        self.projection = nn.Sequential(
            nn.Linear(self.embed_dims, 128),
            nn.BatchNorm1d(num_features=128),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )
        self.gen = nn.Sequential(
            nn.ConvTranspose2d(self.noise_dim + 128, 64 * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(64 * 8),
            nn.ReLU(True),

            nn.ConvTranspose2d(64 * 8, 64 * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64 * 4),
            nn.ReLU(True),

            nn.ConvTranspose2d(64 * 4, 64 * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64 * 2),
            nn.ReLU(True),

            nn.ConvTranspose2d(64 * 2, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            nn.ConvTranspose2d(64, self.inp_channels, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, embed, z):
        project_embed = self.projection(embed).view(-1, 128, 1, 1)
        return self.gen(torch.cat([project_embed, z], 1))

class Discriminator(nn.Module):
    def __init__(self, image_size=64, inp_channels=3, projected_dims=128, embed_dims=1024):
        super().__init__()
        self.image_size = image_size
        self.inp_channels = inp_channels
        self.projected_dims = projected_dims
        self.embed_dims = embed_dims

        self.dis1 = nn.Sequential(
            nn.Conv2d(self.inp_channels, 64, 4,2,1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 64*2, 4,2,1, bias=False),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.Conv2d(64*2, 64*4, 4,2,1, bias=False),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.Conv2d(64*4, 64*8, 4,2,1, bias=False),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )

        self.projection = nn.Sequential(
            nn.Linear(self.embed_dims, self.projected_dims),
            nn.BatchNorm1d(num_features=self.projected_dims),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )

        self.dis2 = nn.Conv2d(64*8+self.projected_dims, 1, 4, 1, 0, bias=False)

    def forward(self, inp, embed):
        d1 = self.dis1(inp)
        proj_em = self.projection(embed).repeat(4,4,1,1).permute(2,3,0,1)
        return self.dis2(torch.cat([d1, proj_em], 1)).view(-1, 1).squeeze(1)



if __name__ == '__main__':
    gen = Generator()
    em = torch.rand(14, 1024)
    z = torch.rand(14, 100, 1, 1)
    img = gen(em, z)
    assert img.shape == torch.Size([14, 3, 64, 64])
    print("Success")

    dis = Discriminator()
    print(dis(img, em).shape)

