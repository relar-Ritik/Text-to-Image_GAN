import torch
from torch import nn
from tqdm import trange
from models import vanilla_gan, cgan
import os

class DCGAN(object):

    def __init__(self, epochs, batch_size, device, save_path, num_classes, G_type = "vanilla_gan", D_type = "vanilla_gan",
                 lr=0.0002, d_beta1 =0.5, d_beta2= 0.999, g_beta1 =0.5, g_beta2= 0.999, embed_size = 100):

        self.embed_size = embed_size
        self.num_classes = num_classes
        self.G_type = G_type
        self.D_type =D_type
        self.lr = lr
        self.d_beta1 = d_beta1
        self.d_beta2 =d_beta2
        self.g_beta1 = g_beta1
        self.g_beta2 = g_beta2
        self.device = device

        if self.G_type == "vanilla_gan":
            self.G = vanilla_gan.Generator().to(device)
        elif self.G_type == "cgan":
            self.G = cgan.Generator(embed_size=embed_size, num_classes=num_classes).to(device)

        if self.D_type == "vanilla_gan":
            self.D = vanilla_gan.Discriminator().to(device)
        elif self.D_type == "cgan":
            self.D = cgan.Discriminator(embed_size=embed_size, num_classes=num_classes).to(device)

        self.loss = nn.BCELoss()

        self.d_optimizer = torch.optim.Adam(self.D.parameters(), lr=lr, betas=(d_beta1, d_beta2))
        self.g_optimizer = torch.optim.Adam(self.G.parameters(), lr=lr, betas=(g_beta1, g_beta2))

        self.epochs = epochs
        self.batch_size = batch_size

        self.number_of_images = 10
        self.save_path = save_path

    def train(self, train_loader):
        disc_loss = []
        genr_loss = []

        generator_iter = 0

        for epoch in trange(self.epochs):

            for i, batch_data in enumerate(train_loader):
                # Step 1: Train discriminator
                images = batch_data['right_images']
                z = torch.rand((images.size(0), 100, 1, 1))

                real_labels = torch.ones(images.size(0))
                fake_labels = torch.zeros(images.size(0))

                images, z = images.to(self.device), z.to(self.device)
                real_labels, fake_labels = real_labels.to(self.device), fake_labels.to(self.device)

                # Compute the BCE Loss using real images
                real_logits = self.D(images)
                real_logits = torch.squeeze(real_logits)
                d_loss_real = self.loss(real_logits, real_labels)

                # Compute the BCE Loss using fake images
                fake_images = self.G(z)
                fake_logits = self.D(fake_images)
                fake_logits = torch.squeeze(fake_logits)
                d_loss_fake = self.loss(fake_logits, fake_labels)

                # Optimize discriminator
                d_loss = d_loss_real + d_loss_fake
                self.D.zero_grad()
                d_loss.backward()
                self.d_optimizer.step()

                # Step 2: Train Generator
                z = torch.randn(images.size(0), 100, 1, 1).to(self.device)

                fake_images = self.G(z)
                fake_logits = self.D(fake_images)
                fake_logits = torch.squeeze(fake_logits)
                g_loss = self.loss(fake_logits, real_labels)

                self.D.zero_grad()
                self.G.zero_grad()
                g_loss.backward()
                self.g_optimizer.step()
                generator_iter += 1

                disc_loss.append(d_loss.item())
                genr_loss.append(g_loss.item())

            if epoch % 10 == 0:
                torch.save(self.G.state_dict(),
                           os.path.join(self.save_path, 'G_ckpt_{}_{}.pth'.format(epoch, self.G_type)))
                torch.save(self.D.state_dict(),
                           os.path.join(self.save_path, 'D_ckpt_{}_{}.pth'.format(epoch, self.D_type)))

        return disc_loss, genr_loss

    def generate_img(self, z, number_of_images):
        samples = self.G(z).data.cpu().numpy()[:number_of_images]
        generated_images = []
        for sample in samples:
            generated_images.append(sample.reshape(3, 64, 64).transpose(1, 2, 0))
        return generated_images