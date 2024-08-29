import numpy as np
import torch
from torchvision.datasets import MovingMNIST
from torch.utils.data import DataLoader, Dataset, Subset
import matplotlib.pyplot as plt  # Add this import for visualization
from test_flow import train_flow_matching, generate_trajectory
from unet import UNet

class MovingMNISTFlowMatchingDataset(Dataset):
    def __init__(self, subset_size=200, num_samples=10, noise_std=0.1, dt=0.1, denoising=False):
        """
        Args:
            subset_size (int): Number of samples to use from the MovingMNIST dataset.
            num_samples (int): Number of samples in each trajectory.
            noise_std (float): Standard deviation of Gaussian noise added to the trajectory.
            dt (float): Time step between samples.
            denoising (bool): Whether to apply denoising to the input data.
        """
        self.noise_std = noise_std
        self.denoising = denoising
        self.data = []  # Flattened list of all trajectory segments

        # Load the MovingMNIST dataset
        dataset = MovingMNIST(root='./data', split='test', download=True)

        # Create a small sub-sample of the dataset
        subset_indices = list(range(subset_size))
        subset = Subset(dataset, subset_indices)
        dataloader = DataLoader(subset, batch_size=1, shuffle=False)

        trajectories = []

        for data in dataloader:
            data = data.squeeze(0).numpy()  # Remove batch dimension and convert to numpy array
            num_frames = data.shape[0]

            for i in range(num_frames - num_samples + 1):
                trajectory = []
                for j in range(num_samples):
                    t = j * dt
                    frame = data[i + j].flatten()  # Flatten the image frame
                    trajectory.append(np.concatenate(([t], frame)))
                trajectories.append(np.array(trajectory))

        # Generate noisy data and other required tensors
        for trajectory in trajectories:
            for idx in range(len(trajectory) - 1):
                t_start, *x_start = trajectory[idx]
                t_end, *x_end = trajectory[idx + 1]

                x_start = np.array(x_start).reshape(1, 64, 64).astype(np.float32)  # Ensure float32
                x_end = np.array(x_end).reshape(1, 64, 64).astype(np.float32)  # Ensure float32

                # Interpolate position
                t_rand = np.random.uniform(t_start, t_end)
                x_interp = (1 - (t_rand - t_start) / (t_end - t_start)) * x_start + ((t_rand - t_start) / (t_end - t_start)) * x_end

                # Calculate dx/dt
                dx_dt = (x_end - x_start) / (t_end - t_start)

                # Add synthetic Gaussian noise
                noise = np.random.normal(0, noise_std, x_interp.shape).astype(np.float32)  # Ensure float32
                noisy_data = x_interp + noise

                clean_data = torch.tensor(x_interp, dtype=torch.float32)
                target_data = torch.tensor(dx_dt, dtype=torch.float32)
                delta_t_tensor = torch.tensor(t_end - t_start, dtype=torch.float32)

                self.data.append((noisy_data, clean_data, target_data, -noise, delta_t_tensor))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.denoising:
            return self.data[idx]
        else:
            _, clean_data, target_data, _, delta_t_tensor = self.data[idx]
            return clean_data, target_data, delta_t_tensor

# Example usage
dataset = MovingMNISTFlowMatchingDataset(subset_size=20, num_samples=10, noise_std=0.1, dt=0.1, denoising=True)

# Visualize one sample
noisy_data, clean_data, target_data, noise, delta_t_tensor = dataset[0]

# Print min and max values
print("Noisy Data - Min:", np.min(noisy_data), "Max:", np.max(noisy_data))
print("Clean Data - Min:", torch.min(clean_data).item(), "Max:", torch.max(clean_data).item())
print("Target Data - Min:", torch.min(target_data).item(), "Max:", torch.max(target_data).item())

# Reshape the data to 64x64 images
noisy_image = noisy_data.reshape(64, 64)
clean_image = clean_data.numpy().reshape(64, 64)
target_image = target_data.numpy().reshape(64, 64)  # Assuming target_data can be reshaped similarly

# Plot the images
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.title("Noisy Image")
plt.imshow(noisy_image, cmap='gray')

plt.subplot(1, 3, 2)
plt.title("Clean Image")
plt.imshow(clean_image, cmap='gray')

plt.subplot(1, 3, 3)
plt.title("Target Data")
plt.imshow(target_image, cmap='gray')

plt.show()

# Print the shape of the first processed data to verify
print(f"Length of dataset: {len(dataset)}")
print(f"Shape of first processed data: {dataset[0][0].shape}")  # Should be (1, 64, 64) for 64x64 images

# Train the SDE model using the MovingMNIST dataset
trained_model, denoiser, diffusion_model = train_flow_matching(
        dataset=dataset,
        flow_backbone=lambda: UNet(in_channels=1, out_channels=1, features=8, use_tanh=False),
        diffusion_backbone=lambda min_val, max_val: UNet(in_channels=1, out_channels=1, features=8, use_tanh=True, min_val=min_val, max_val=max_val),
        num_epochs=30,
        batch_size=4,
        learning_rate=0.001,
        denoising=True,
        l2_reg=0.0,
        diffusion_min=-10.0,
        diffusion_max=0.0,
        initial_epochs_for_flow_only=100
    )

print("Training completed.")

# Generate a sample time sequence for the MovingMNIST sequence prediction
initial_point = clean_data.numpy().flatten()  # Flatten the initial point
t_values = np.linspace(0, 1, 10)  # Example time values

generated_trajectory = generate_trajectory(
    model=trained_model,
    diffusion_model=diffusion_model,
    initial_point=initial_point,
    t_values=t_values,
    denoiser=denoiser,
    rtol=1e-2,
    atol=1e-4,
    use_sde=False,
    denoising_magnitude=2.0,
    noise_std=0.1,
    is_image=True  # Set the flag to True for image input
)

# Ensure the reshape dimensions match the number of elements
generated_images = generated_trajectory.reshape(-1, 64, 64)

# Plot the generated images
plt.figure(figsize=(15, 5))
for i, img in enumerate(generated_images):
    plt.subplot(1, len(generated_images), i + 1)
    plt.title(f"Generated Image {i + 1}")
    plt.imshow(img, cmap='gray')
plt.show()