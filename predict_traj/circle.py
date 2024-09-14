import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
from methods import langvin_sample, energy_discrepancy
import os
import torch.nn as nn

# Set CUDA launch blocking to avoid race conditions
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

class EBM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(EBM, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.act1 = nn.SiLU()
        self.fc2 = nn.Linear(hidden_size,hidden_size)
        self.act2 = nn.SiLU()
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.act3 = nn.SiLU()
        self.fc4 = nn.Linear(hidden_size, output_size)
        
    def forward(self, act, obs):
        x = torch.cat((act.float(), obs.float()), dim=-1)
        x = self.act1(self.fc1(x))
        x = self.act2(self.fc2(x))
        x = self.act3(self.fc3(x))
        x = self.fc4(x)
        return x
# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define the function to train the EBM model
def train_ebm(X, Y, t_noise=0.3, num_epochs=2000, batch_size=32, hidden_size=512):
    X_tensor = torch.tensor(X, dtype=torch.float32)
    Y_tensor = torch.tensor(Y, dtype=torch.float32)
    dataset = TensorDataset(X_tensor, Y_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    model = EBM(X.shape[-1]+Y.shape[-1], hidden_size, 1).to(device)
    optimizer = optim.Adam(model.parameters())
    
    for epoch in range(num_epochs):
        for input, target in dataloader:
            input = input.to(device)
            target = target.to(device)
            loss, _, _  = energy_discrepancy(model, target, input, t_noise=t_noise, m_particles=256, w_stable=0.5)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if epoch % 100 == 0:
            print(f'Epoch [{epoch}/{num_epochs}], Loss: {loss.item():.4f}')
    
    return model

# Function to generate the trajectory step by step
def generate_trajectory_ebm(model, A, n_steps):
    trajectory = [A]  # Start with point A
    current_point = torch.tensor(A, dtype=torch.float32).to(device)
    
    model.eval()
    x, y = torch.linspace(-4., 4., 1000), torch.linspace(-4., 4., 1000)
    X_mesh, Y_mesh = torch.meshgrid(x, y)
    X_mesh = X_mesh[...,None]
    Y_mesh = Y_mesh[...,None]
    p = torch.cat((X_mesh.to(device), Y_mesh.to(device)), dim=-1)
    for _ in range(n_steps):
        temp = current_point[None, None, ...].repeat([1000, 1000, 1])
        Z_mesh = ebm(p, temp.to(device)).cpu().detach().numpy()
        i, j, _ = np.unravel_index(np.argmin(Z_mesh), Z_mesh.shape)
        trajectory.append(np.array([x[i], y[j]]))
        # Predict the next two points using the current point
        '''next_points = langvin_sample(model, current_point, current_point).cpu().detach().numpy()
        
        # Add only the first predicted point to the trajectory
        first_predicted_point = next_points[0]  # Get the first point in the prediction
        trajectory.append(first_predicted_point)
        '''
        # Update the current point for the next prediction
        current_point = torch.tensor(([x[i], y[j]]), dtype=torch.float32).to(device)
    
    return np.array(trajectory)

# Function to create a broken line (piecewise linear path)
def create_broken_line(points, n_per_segment):
    X, Y = [], []
    for i in range(len(points) - 1):
        A = points[i]
        B = points[i + 1]
        t = np.random.uniform(0, 1, n_per_segment)
        segment_X = (1 - t)[:, np.newaxis] * A + t[:, np.newaxis] * B
        c = (B - A) / n_per_segment
        segment_Y = segment_X + c
        X.append(segment_X)
        Y.append(segment_Y)
    return np.vstack(X), np.vstack(Y)

def simulate_trajectory(num_samples, amplitude=1.0, freq=1.0, random_sampling=False):
    X, Y = [], []

    if random_sampling:
        for _ in range(num_samples):
            x = np.random.uniform(0, 2 * np.pi)
            y = amplitude * np.sin(freq * x)
            dt = 0.1 #np.random.uniform(0.09, 0.11)
            x_next = x + dt
            y_next = amplitude * np.sin(freq * x_next)
            X.append(np.array([x, y]))
            Y.append(np.array([x_next, y_next]))
        X = np.array(X)
        Y = np.array(Y)
    else:
        t = np.linspace(0, 2 * np.pi, num_samples + 1)[:-1]
        x = t
        y = amplitude * np.sin(freq * t)
        x_next = t + 0.1
        y_next = amplitude * np.sin(freq * x_next)
        X = np.stack((x, y), axis=-1)
        Y = np.stack((x_next, y_next), axis=-1)

    return X, Y

def simulate_circle(num_samples, radius=1.0, random_sampling=False):
    X, Y = [], []
    if random_sampling:
        for _ in range(num_samples):
            theta = np.random.uniform(0, 2 * np.pi)
            x = radius * np.cos(theta)
            y = radius * np.sin(theta)
            dtheta = 2*np.pi/num_samples #np.random.uniform(0.09, 0.11)
            x_next = radius * np.cos(theta+dtheta)
            y_next = radius * np.sin(theta+dtheta)
            X.append(np.array([x, y]))
            Y.append(np.array([x_next, y_next]))
        X = np.array(X)
        Y = np.array(Y)
    else:
        theta = np.linspace(0, 2 * np.pi, num_samples + 1)[:-1]
        x = radius * np.cos(theta)
        y = radius * np.sin(theta)
        dtheta = 2*np.pi/num_samples #np.random.uniform(0.09, 0.11)
        x_next = radius * np.cos(theta+dtheta)
        y_next = radius * np.sin(theta+dtheta)
        X = np.stack((x, y), axis=-1)
        Y = np.stack((x_next, y_next), axis=-1)
    return X, Y

X, Y = simulate_circle(20, radius=3, random_sampling=True)

# Plot the original broken line
# Define MLP for trajectory prediction
class TrajectoryMLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(TrajectoryMLP, self).__init__()
        self.layers = nn.ModuleList()
        
        # Input layer
        self.layers.append(nn.Linear(input_size, hidden_sizes[0]))
        self.layers.append(nn.ReLU())
        
        # Hidden layers
        for i in range(1, len(hidden_sizes)):
            self.layers.append(nn.Linear(hidden_sizes[i-1], hidden_sizes[i]))
            self.layers.append(nn.ReLU())
        
        # Output layer
        self.layers.append(nn.Linear(hidden_sizes[-1], output_size))
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

# Train MLP
def train_mlp(X, Y, epochs=1000):
    model = TrajectoryMLP(2, [128, 512, 512, 128], 2).to(device)
    optimizer = optim.Adam(model.parameters())
    criterion = nn.MSELoss()
    
    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
    Y_tensor = torch.tensor(Y, dtype=torch.float32).to(device)
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X_tensor)
        loss = criterion(outputs, Y_tensor)
        loss.backward()
        optimizer.step()
    
    return model

# Generate trajectory using MLP
def generate_trajectory_mlp(model, start_point, num_steps):
    trajectory = [start_point]
    current_point = torch.tensor(start_point, dtype=torch.float32).unsqueeze(0).to(device)
    
    for _ in range(num_steps):
        next_point = model(current_point)
        trajectory.append(next_point.squeeze().cpu().detach().numpy())
        current_point = next_point
    
    return np.array(trajectory)

# Function to plot arrow map
def plot_arrow_map(ax, model, x_range, y_range, density=20):
    x = np.linspace(x_range[0], x_range[1], density)
    y = np.linspace(y_range[0], y_range[1], density)
    X, Y = np.meshgrid(x, y)
    
    U = np.zeros_like(X)
    V = np.zeros_like(Y)
    
    for i in range(density):
        for j in range(density):
            input_tensor = torch.tensor([[X[i, j], Y[i, j]]], dtype=torch.float32).to(device)
            output = model(input_tensor).squeeze().cpu().detach().numpy()
            U[i, j] = output[0] - X[i, j]
            V[i, j] = output[1] - Y[i, j]
    
    ax.quiver(X, Y, U, V, scale=30, width=0.002)

def plot_arrow_map_ebm(ax, ebm, x_range, y_range, density=20):
    x = np.linspace(x_range[0], x_range[1], density)
    y = np.linspace(y_range[0], y_range[1], density)
    X, Y = np.meshgrid(x, y)
    
    U = np.zeros_like(X)
    V = np.zeros_like(Y)
    
    for i in range(density):
        for j in range(density):
            input_point = torch.tensor([[X[i, j], Y[i, j]]], dtype=torch.float32).to(device)
            
            output = langvin_sample(ebm, input_point, input_point, e_l_step_size=0.02, noise_scale=0.1)[0].cpu().detach().numpy()
            
            U[i, j] = output[0] - X[i, j]
            V[i, j] = output[1] - Y[i, j]
    
    ax.quiver(X, Y, U, V, scale=30, width=0.002)
    
# Plot original data and MLP trajectories
fig, ax = plt.subplots(1, 2, figsize=(16, 8))

# Train MLP and EBM models
mlp = train_mlp(X, Y)
ebm = train_ebm(X, Y)

# Generate trajectories
mlp_trajectory = generate_trajectory_mlp(mlp, np.array([-3., 0.0]), len(X))
ebm_trajectory = generate_trajectory_ebm(ebm, np.array([-3., 0.0]), len(X))

# Plot MLP results
ax[0].scatter(X[:, 0], X[:, 1], c='blue', alpha=0.5, label='Original Data')
ax[0].plot(mlp_trajectory[:, 0], mlp_trajectory[:, 1], c='orange', label='MLP Trajectory')
plot_arrow_map(ax[0], mlp, x_range=(-4, 4), y_range=(-4, 4))
ax[0].legend()
ax[0].grid(True)
ax[0].set_title('MLP Trajectory and Arrow Map')

# Plot EBM results
ax[1].scatter(X[:, 0], X[:, 1], c='blue', alpha=0.5, label='Original Data')
ax[1].plot(ebm_trajectory[:, 0], ebm_trajectory[:, 1], c='green', label='EBM Trajectory')
plot_arrow_map_ebm(ax[1], ebm, x_range=(-4, 4), y_range=(-4, 4))
ax[1].legend()
ax[1].grid(True)
ax[1].set_title('EBM Trajectory and Arrow Map')

plt.tight_layout()
plt.show()
plt.savefig('predict_circle_comparison_mlp_ebm.png')