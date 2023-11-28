# Import libraries
import numpy as np
import torch
import yaml
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from PIL import Image
import os
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import  random

from Dataset import T2IGANDataset
from models.dcgan import DCGAN

# Configuration setup
config_path = 'config.yaml'
with open(config_path, 'r') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

# Global parameters
batch_size = 64
lr = 0.0002
epochs = 1
num_channels=3
G_type = "cgan" # Generator type
D_type = "cgan" # Discriminator type
d_beta1 =0.5
d_beta2= 0.999
g_beta1 =0.5
g_beta2= 0.999
save_path = 'ckpt'
dataset = T2IGANDataset(dataset_file="data/flowers.hdf5", split="train")
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
embed_size = 1024 # if using cGAN

# Set device to GPU
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

# Set seed for reproducibility
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

# Training
model = DCGAN(epochs, batch_size, device, save_path, G_type , D_type ,
              lr, d_beta1 , d_beta2, g_beta1, g_beta2, embed_size)
disc_loss, genr_loss = model.train(train_loader)

# Plot the generated images
z = torch.randn(100, 100, 1, 1).to(device)
generated_images = model.generate_img(z, 100, train_loader=train_loader)
fig = plt.figure(figsize=(10, 10))
gs = gridspec.GridSpec(10, 10)
gs.update(wspace=0.05, hspace=0.05)

for i in range(10):
    for j in range(10):
        sample = generated_images[i * 10 + j]
        ax = plt.subplot(gs[i, j])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')

        image = sample.reshape(64, 64, 3)
        image = (image - image.min()) / (image.max() - image.min())
        plt.imshow(image)

plt.savefig("fig/generated_images_{}_{}.png".format(G_type, D_type), bbox_inches='tight')
plt.show()

# Plot the discriminator and generator loss.
def plot_gan_losses(disc_loss, genr_loss):
    fig = plt.figure(figsize=(20, 8))
    fig.add_subplot(121)
    plt.title('Discriminator Loss', fontsize=16)
    plt.xlabel('Iteration', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt1 = plt.plot(disc_loss)
    fig.add_subplot(122)
    plt.title('Generator Loss', fontsize=16)
    plt.xlabel('Iteration', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt2 = plt.plot(genr_loss)

    # Save the figure
    plt.savefig("fig/gan_losses_{}_{}.png".format(G_type, D_type), bbox_inches='tight')
    plt.show()

plot_gan_losses(disc_loss, genr_loss)
