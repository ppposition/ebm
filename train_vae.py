import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import MovingMNIST
from typing import List
import numpy as np
import matplotlib.pyplot as plt
from vae_model import VanillaVAE  # Assuming the VAE model is saved in vae_model.py

# Hyperparameters
batch_size = 64
learning_rate = 1e-3
num_epochs = 20
latent_dim = 128
in_channels = 1  # For Moving MNIST dataset

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Custom Dataset for Moving MNIST
class MovingMNISTImageDataset(Dataset):
    def __init__(self, root, train, transform=None, subset_size=None):
        self.dataset = MovingMNIST(
            root=root,
            split="train" if train else "test",
            download=True,
            transform=transform,
        )
        (
            self.num_sequences,
            self.sequence_length,
            self.num_channels,
            self.height,
            self.width,
        ) = self.dataset.data.shape

        if subset_size:
            self.num_sequences = min(self.num_sequences, subset_size)
            self.dataset.data = self.dataset.data[:self.num_sequences]

    def __len__(self):
        return self.num_sequences * self.sequence_length

    def __getitem__(self, idx):
        sequence_idx = idx // self.sequence_length
        frame_idx = idx % self.sequence_length
        sample = self.dataset.data[sequence_idx, frame_idx, :, :, :]
        if self.dataset.transform:
            sample = self.dataset.transform(sample)
        return sample


# Data loading with subset
subset_size = 5  # Define the size of the subset
train_dataset = MovingMNISTImageDataset(root="./data", train=False, subset_size=subset_size)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Model, optimizer, and loss function
model = VanillaVAE(in_channels=in_channels, latent_dim=latent_dim, hidden_dims=[32, 64], input_height=64, input_width=64).to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


# Training loop
def train(model, train_loader, optimizer, num_epochs):
    model.train()
    for epoch in range(num_epochs):
        for batch_idx, data in enumerate(train_loader):
            data = data.float().to(device)  # Convert input to float and move to device
            optimizer.zero_grad()
            recon_batch, mu, log_var = model(data)
            loss_dict = model.loss_function(recon_batch, data, mu, log_var)
            loss = loss_dict["loss"]
            loss.backward()
            optimizer.step()
            if batch_idx % 1 == 0:
                print(loss.item())
                print(
                    f"Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}"
                )


# Run training
train(model, train_loader, optimizer, num_epochs)

# Visualization of interpolation
def visualize_interpolation(model, dataset, num_steps=10):
    model.eval()
    with torch.no_grad():
        # Get two random images from the dataset
        idx1, idx2 = np.random.randint(0, len(dataset), size=2)
        x1 = dataset[idx1].unsqueeze(0).float().to(device)
        x2 = dataset[idx2].unsqueeze(0).float().to(device)
        
        # Visualize interpolation
        interpolations = model.interpolate(x1, x2, num_steps)
        
        # Plot the original images and interpolations
        fig, axes = plt.subplots(1, num_steps + 2, figsize=(15, 2))
        axes[0].imshow(x1.squeeze().cpu(), cmap='gray')
        axes[0].set_title('Original 1')
        axes[0].axis('off')
        
        for i, img in enumerate(interpolations):
            axes[i + 1].imshow(img.squeeze(), cmap='gray')
            axes[i + 1].axis('off')
        
        axes[-1].imshow(x2.squeeze().cpu(), cmap='gray')
        axes[-1].set_title('Original 2')
        axes[-1].axis('off')
        
        plt.show()

# Visualize interpolation after training
visualize_interpolation(model, train_dataset, num_steps=10)
