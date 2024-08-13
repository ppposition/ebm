import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

# Function to generate synthetic data with different dimensions
def generate_synthetic_data(n_samples, data_dim):
    X = np.linspace(-np.pi, np.pi, n_samples)
    Y = np.sin(X)
    # Create higher-dimensional data by adding zero-valued features
    data = np.zeros((n_samples, data_dim))
    data[:, 0] = X
    data[:, 1] = Y

    # Apply a random rotation
    rotation_matrix = np.random.randn(data_dim, data_dim)
    q, _ = np.linalg.qr(rotation_matrix)  # QR decomposition to ensure orthogonality
    data = np.dot(data, q)
    
    return data

# Custom dataset that adds noise dynamically
class NoisyDataset(Dataset):
    def __init__(self, data, noise_factor=0.2):
        self.data = data
        self.noise_factor = noise_factor

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        clean_data = self.data[idx]
        noise = self.noise_factor * np.random.normal(loc=0.0, scale=1.0, size=clean_data.shape)
        noisy_data = clean_data + noise
        return torch.tensor(noisy_data, dtype=torch.float32), torch.tensor(clean_data, dtype=torch.float32)

# Define the autoencoder model
class Autoencoder(nn.Module):
    def __init__(self, input_dim):
        super(Autoencoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 2)  # Bottleneck layer
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# Function to train the autoencoder
def train_autoencoder(data, noise_factor=0.2, num_epochs=100):
    data_tensor = torch.tensor(data, dtype=torch.float32)

    # Create DataLoader
    dataset = NoisyDataset(data_tensor, noise_factor=noise_factor)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Initialize model, loss function, and optimizer
    model = Autoencoder(input_dim=data.shape[1])
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train the model
    for epoch in range(num_epochs):
        for noisy_data, clean_data in dataloader:
            # Forward pass
            output = model(noisy_data)
            loss = criterion(output, clean_data)
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

    return model

# Function to evaluate the autoencoder with different noise levels
def evaluate_autoencoder(model, data, noise_factor=0.2):
    # Create noisy test data
    noise = noise_factor * np.random.normal(loc=0.0, scale=1.0, size=data.shape)
    noisy_data_tensor = torch.tensor(data + noise, dtype=torch.float32)
    clean_data_tensor = torch.tensor(data, dtype=torch.float32)

    # Evaluate the model
    with torch.no_grad():
        denoised_data_tensor = model(noisy_data_tensor)
        mse = nn.MSELoss()(denoised_data_tensor, clean_data_tensor).item()

    return noisy_data_tensor.numpy(), denoised_data_tensor.numpy(), mse

# Test on different data dimensions and noise levels
data_dimensions = [2, 5, 10, 50, 100]
train_noise_factor = 0.2
test_noise_factors = [0.1, 0.2, 0.3]
results = {}

for data_dim in data_dimensions:
    print(f"Training on data dimension: {data_dim}")
    data = generate_synthetic_data(1000, data_dim)  # Generate fixed clean data for both training and testing
    model = train_autoencoder(data, noise_factor=train_noise_factor)

    results[data_dim] = {}
    for test_noise_factor in test_noise_factors:
        print(f"Testing on data dimension {data_dim} with noise factor {test_noise_factor}")
        noisy_data, denoised_data, mse = evaluate_autoencoder(model, data, noise_factor=test_noise_factor)
        results[data_dim][test_noise_factor] = mse
        print(f"MSE for dimension {data_dim} with noise factor {test_noise_factor}: {mse:.4f}")

# Plot for the 2D case
data_dim = 2
print(f"Plotting for data dimension: {data_dim}")
data = generate_synthetic_data(1000, data_dim)  # Generate fixed clean data for plotting
model = train_autoencoder(data, noise_factor=train_noise_factor)

plt.figure(figsize=(15, 5 * len(test_noise_factors)))

for i, test_noise_factor in enumerate(test_noise_factors):
    noisy_data, denoised_data, mse = evaluate_autoencoder(model, data, noise_factor=test_noise_factor)

    plt.subplot(len(test_noise_factors), 1, i + 1)
    plt.scatter(data[:, 0], data[:, 1], color='blue', s=2, label='Original Data')
    plt.scatter(noisy_data[:, 0], noisy_data[:, 1], color='red', s=2, label=f'Noisy Data (Noise={test_noise_factor})')
    plt.scatter(denoised_data[:, 0], denoised_data[:, 1], color='green', s=2, label='Denoised Data')
    plt.title(f'Original, Noisy and Denoised Data (Noise={test_noise_factor})')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()

plt.tight_layout()
plt.show()