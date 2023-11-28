import torch
from torch import nn
from tqdm import trange
from models import vanilla_gan, cgan
import os

class DCGAN(object):

    def __init__(self, epochs, batch_size, device, save_path, G_type = "vanilla_gan", D_type = "vanilla_gan",
                 lr=0.0002, d_beta1 =0.5, d_beta2= 0.999, g_beta1 =0.5, g_beta2= 0.999, embed_size = 100):

        self.embed_size = embed_size
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
            self.G = cgan.Generator(embed_size=embed_size).to(device)

        if self.D_type == "vanilla_gan":
            self.D = vanilla_gan.Discriminator().to(device)
        elif self.D_type == "cgan":
            self.D = cgan.Discriminator(embed_size=embed_size).to(device)

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

        for epoch in trange(self.epochs):
            for i, batch_data in enumerate(train_loader):
                # Load data
                right_images = batch_data['right_images'].to(self.device)
                z = torch.randn(right_images.size(0), 100, 1, 1).to(self.device)
                real_labels = torch.ones(right_images.size(0)).to(self.device)
                fake_labels = torch.zeros(right_images.size(0)).to(self.device)

                # Conditional inputs
                right_embed = None
                wrong_images = None
                if self.D_type == "cgan":
                    right_embed = batch_data['right_embed'].to(self.device)
                    wrong_images = batch_data['wrong_images'].to(self.device)

                # Generate fake images
                fake_images = None
                if self.G_type == "vanilla_gan":
                    fake_images = self.G(z)
                elif self.G_type == "cgan":
                    fake_images = self.G(z, right_embed)

                # Train Discriminator
                self.D.zero_grad()

                if self.D_type == "vanilla_gan":
                    # Real images
                    real_logits = self.D(right_images)
                    # Fake images
                    fake_logits = self.D(fake_images)

                elif self.D_type == "cgan":
                    # Real images
                    real_logits = self.D(right_images, right_embed)[0]
                    # Fake images
                    fake_logits = self.D(fake_images, right_embed)[0]

                # Discriminator losses
                d_loss_real = self.loss(real_logits.squeeze(), real_labels)
                d_loss_fake = self.loss(fake_logits.squeeze(), fake_labels)

                # Wrong image/text losses for cGAN
                d_loss_wrong = 0
                if self.D_type == "cgan" and wrong_images is not None:
                    wrong_logits = self.D(wrong_images, right_embed)[0].squeeze()
                    d_loss_wrong = self.loss(wrong_logits, fake_labels)

                # Total discriminator loss
                d_loss = d_loss_real + d_loss_fake + d_loss_wrong
                d_loss.backward()
                self.d_optimizer.step()

                # Regenerate fake images for Generator's backward pass
                z = torch.randn(right_images.size(0), 100, 1, 1).to(self.device)
                fake_images = self.G(z, right_embed) if self.D_type == "cgan" else self.G(z)

                # Train Generator
                self.G.zero_grad()
                if self.D_type == "vanilla_gan":
                    fake_logits = self.D(fake_images)
                elif self.D_type == "cgan":
                    fake_logits = self.D(fake_images, right_embed)[0]
                g_loss = self.loss(fake_logits.squeeze(), real_labels)
                g_loss.backward()
                self.g_optimizer.step()

                disc_loss.append(d_loss.item())
                genr_loss.append(g_loss.item())

            if epoch % 10 == 0:
                torch.save(self.G.state_dict(),
                           os.path.join(self.save_path, 'G_ckpt_{}_{}.pth'.format(epoch, self.G_type)))
                torch.save(self.D.state_dict(),
                           os.path.join(self.save_path, 'D_ckpt_{}_{}.pth'.format(epoch, self.D_type)))

        return disc_loss, genr_loss

    def generate_img(self, z, number_of_images, train_loader, text_embedding = None):
        if self.G_type == "vanilla_gan":
            samples = self.G(z).data.cpu().numpy()[:number_of_images]
        if self.G_type == "cgan":
            if text_embedding is None:
                batch_data = next(iter(train_loader))
                text_embedding = batch_data['right_embed'].to(self.device)[0:1]  # Taking one embedding
                text_embedding = text_embedding.repeat(z.size(0), 1)  # Repeat to match batch size
            samples = self.G(z, text_embedding).data.cpu().numpy()[:number_of_images]

        generated_images = []
        for sample in samples:
            generated_images.append(sample.reshape(3, 64, 64).transpose(1, 2, 0))
        return generated_images

    def get_random_embedding(self, train_loader):
        random_batch = next(iter(train_loader))
        random_idx = torch.randint(0, random_batch['right_embed'].size(0), (1,))
        return random_batch['right_embed'][random_idx].to(self.device)