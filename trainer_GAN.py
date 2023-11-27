# Import libraries
import numpy as np
import torch
import yaml
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from PIL import Image
import os

from Dataset import T2IGANDataset
from models.dcgan import DCGAN
#from utils import Utils, Logger

# Configuration setup
config_path = 'config.yaml'
with open(config_path, 'r') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

# Global parameters
batch_size = 64
lr = 0.0002
epochs = 1
num_channels=3
G_type = "vanilla_gan" # Generator type
D_type = "vanilla_gan" # Discriminator type
d_beta1 =0.5
d_beta2= 0.999
g_beta1 =0.5
g_beta2= 0.999
save_path = '.ckpt'
dataset = T2IGANDataset(dataset_file="data/flowers.hdf5", split="train")
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Set device to GPU
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

# Training
model = DCGAN(epochs, batch_size,  device, G_type , D_type , lr, d_beta1 , d_beta2, g_beta1, g_beta2)
disc_loss, genr_loss = model.train(train_loader)