import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
from methods import langvin_sample, energy_discrepancy
import os
import torch.nn as nn
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
    
def create_combined_trajectory(num_points_line=100, num_points_semiellipse=80, num_points_convergence=30):
    # Parameters for the trajectory
    line_start = np.array([0.0, 0.0])
    line_end = np.array([1.0, 0.0])
    semiellipse_a = 2.0  # Semi-major axis
    semiellipse_b = 2.0  # Semi-minor axis
    convergence_point = np.array([4.0, 0.0])
    
    # 1. Create the straight line segment with random sampling
    t_line = np.sort(np.random.rand(num_points_line))
    line_segment = np.outer(1 - t_line, line_start) + np.outer(t_line, line_end)
    
    # Adjust y values based on condition
    line_segment_y = line_segment.copy()
    line_segment_y[:, 0] += 0.1
    condition = line_segment_y[:, 0] > 1
    y_values = line_segment_y[condition, 0]-1.0
    random_sign = np.random.rand(sum(condition)) > 0.5
    line_segment_y[condition, 1] = np.where(random_sign, y_values, -y_values)
    line_segment_y[condition, 0] = 1.0

    # 2. Create the top semi-ellipse with random sampling and rotation
    t_semiellipse_top = np.sort(np.random.rand(num_points_semiellipse)) * np.pi
    semiellipse_top = np.array([line_end[0] + semiellipse_a * (1 - np.cos(t_semiellipse_top)), 
                                semiellipse_b * np.sin(t_semiellipse_top)]).T
    
    # Define the rotation angle (clockwise 10 degrees)
    theta = np.radians(-10)
    rotation_matrix = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)]
    ])
    
    # Rotate the points
    relative_points = semiellipse_top - np.array([3.0, 0.0])
    rotated_points = np.dot(relative_points, rotation_matrix.T) + np.array([3.0, 0.0])
    
    # Adjust y values for y < 0
    condition = rotated_points[:, 1] < 0
    rotated_points[condition, 1] = 0
    rotated_points[condition, 0] = 5.05 - (rotated_points[condition, 0] - 5.0)
    top_y = rotated_points
    
    # 3. Create the bottom semi-ellipse with random sampling and rotation
    t_semiellipse_bottom = np.sort(np.random.rand(num_points_semiellipse)) * np.pi
    semiellipse_bottom = np.array([line_end[0] + semiellipse_a * (1 - np.cos(t_semiellipse_bottom)), 
                                   -semiellipse_b * np.sin(t_semiellipse_bottom)]).T
    
    # Define the rotation angle (counterclockwise 10 degrees)
    theta = np.radians(10)
    rotation_matrix = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)]
    ])
    
    # Rotate the points
    relative_points = semiellipse_bottom - np.array([3.0, 0.0])
    rotated_points = np.dot(relative_points, rotation_matrix.T) + np.array([3.0, 0.0])
    
    # Adjust y values for y > 0
    condition = rotated_points[:, 1] > 0
    rotated_points[condition, 1] = 0
    rotated_points[condition, 0] = 5.05 - (rotated_points[condition, 0] - 5.0)
    bottom_y = rotated_points
    
    # 4. Create the convergence part with random sampling
    t_convergence = np.sort(np.random.rand(num_points_convergence))
    top_convergence = np.outer(1 - t_convergence, np.array([5.0, 0.0])) + np.outer(t_convergence, np.array([6.0, 0.0]))
    convergence_y = top_convergence.copy()
    convergence_y[:, 0] += 0.1
    
    # Combine all segments into one trajectory
    combined_trajectory = np.vstack([line_segment, semiellipse_top, semiellipse_bottom, top_convergence])
    y = np.vstack([line_segment_y, top_y, bottom_y, convergence_y])

    return combined_trajectory, y

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
    current_point = torch.tensor(A, dtype=torch.float32).unsqueeze(0).to(device)
    
    model.eval()
    x, y = torch.linspace(-2., 8., 1000), torch.linspace(-4., 4., 1000)
    X_mesh, Y_mesh = torch.meshgrid(x, y)
    X_mesh = X_mesh[...,None]
    Y_mesh = Y_mesh[...,None]
    p = torch.cat((X_mesh.to(device), Y_mesh.to(device)), dim=-1)
    for _ in range(n_steps):
        '''temp = current_point[None, None, ...].repeat([1000, 1000, 1])
        Z_mesh = ebm(p, temp.to(device)).cpu().detach().numpy()
        i, j, _ = np.unravel_index(np.argmin(Z_mesh), Z_mesh.shape)
        trajectory.append(np.array([x[i], y[j]]))
        current_point = torch.tensor(([x[i], y[j]]), dtype=torch.float32).to(device)'''
        # Predict the next two points using the current point
        next_points = langvin_sample(model, current_point, current_point, e_l_step_size=0.02, noise_scale=0.1)
        current_point = next_points
        trajectory.append(current_point[0].cpu().detach().numpy())
        '''
        # Add only the first predicted point to the trajectory
        first_predicted_point = next_points[0]  # Get the first point in the prediction
        trajectory.append(first_predicted_point)
        '''
        # Update the current point for the next prediction
        
    
    return np.array(trajectory)

# Define MLP for direct trajectory prediction
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

# Function to train the MLP
def train_mlp(X, Y, hidden_sizes=[64, 64], num_epochs=1000, batch_size=32):
    mlp = TrajectoryMLP(2, hidden_sizes, 2).to(device)
    optimizer = optim.Adam(mlp.parameters())
    criterion = nn.MSELoss()
    
    X_tensor = torch.FloatTensor(X).to(device)
    Y_tensor = torch.FloatTensor(Y).to(device)
    dataset = TensorDataset(X_tensor, Y_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    for epoch in range(num_epochs):
        for input, target in dataloader:
            optimizer.zero_grad()
            output = mlp(input)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
        if epoch % 100 == 0:
            print(f'MLP Epoch [{epoch}/{num_epochs}], Loss: {loss.item():.4f}')
    
    return mlp

# Function to generate trajectory using MLP
def generate_trajectory_mlp(mlp, start_point, n_steps, step_size=0.1):
    trajectory = [start_point]
    current_point = torch.tensor(start_point, dtype=torch.float32).unsqueeze(0).to(device)
    
    for _ in range(n_steps):
        next_point = mlp(current_point)
        trajectory.append(next_point.squeeze().cpu().detach().numpy())
        current_point = next_point
    
    return np.array(trajectory)

# Function to plot arrow map
def plot_arrow_map(ax, mlp, x_range, y_range, density=20):
    x = np.linspace(x_range[0], x_range[1], density)
    y = np.linspace(y_range[0], y_range[1], density)
    X, Y = np.meshgrid(x, y)
    
    U = np.zeros_like(X)
    V = np.zeros_like(Y)
    
    for i in range(density):
        for j in range(density):
            input_tensor = torch.tensor([[X[i, j], Y[i, j]]], dtype=torch.float32).to(device)
            output = mlp(input_tensor).squeeze().cpu().detach().numpy()
            U[i, j] = output[0] - X[i, j]
            V[i, j] = output[1] - Y[i, j]
    
    ax.quiver(X, Y, U, V, scale=30, width=0.002)

# Function to plot arrow map for EBM
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


# Train MLP and generate trajectory
fig, ax = plt.subplots(3, 4, figsize=(20, 10))

for i in range(4):
    X, Y = create_combined_trajectory(num_points_line=30, num_points_semiellipse=20, num_points_convergence=20)
    
    # Train and generate EBM trajectory
    ebm = train_ebm(X, Y, batch_size=128)
    ebm_trajectory = generate_trajectory_ebm(ebm, np.array([0.0, 0.0]), 40)
    
    # Train and generate MLP trajectory
    mlp_1 = train_mlp(X, Y)
    mlp_trajectory_1 = generate_trajectory_mlp(mlp_1, np.array([0.0, 0.0]), 40)
    
    mlp_2 = train_mlp(X, Y, hidden_sizes=[128, 512, 128])
    mlp_trajectory_2 = generate_trajectory_mlp(mlp_2, np.array([0.0, 0.0]), 40)
    
    # Plot EBM trajectory
    ax[0, i].plot(ebm_trajectory[:, 0], ebm_trajectory[:, 1], c='orange', label='EBM Trajectory')
    ax[0, i].scatter(X[:, 0], X[:, 1], c='blue', alpha=0.1, label='Training Data')
    plot_arrow_map_ebm(ax[0, i], ebm, x_range=(-1, 7), y_range=(-3, 3))
    ax[0, i].legend()
    ax[0, i].grid(True)
    ax[0, i].set_title(f'EBM Generated Trajectory {i+1}')
    
    # Plot MLP trajectory and arrow map
    ax[1, i].plot(mlp_trajectory_1[:, 0], mlp_trajectory_1[:, 1], c='green', label='MLP Trajectory')
    ax[1, i].scatter(X[:, 0], X[:, 1], c='blue', alpha=0.1, label='Training Data')
    plot_arrow_map(ax[1, i], mlp_1, x_range=(-1, 7), y_range=(-3, 3))
    ax[1, i].legend()
    ax[1, i].grid(True)
    ax[1, i].set_title(f'MLP Generated Trajectory {i+1} with Arrow Map')

    ax[2, i].plot(mlp_trajectory_2[:, 0], mlp_trajectory_2[:, 1], c='green', label='MLP Trajectory')
    ax[2, i].scatter(X[:, 0], X[:, 1], c='blue', alpha=0.1, label='Training Data')
    plot_arrow_map(ax[2, i], mlp_2, x_range=(-1, 7), y_range=(-3, 3))
    ax[2, i].legend()
    ax[2, i].grid(True)
    ax[2, i].set_title(f'MLP Generated Trajectory {i+1} with Arrow Map')
    
plt.tight_layout()
plt.savefig('trajectory_comparison_ellipse_4x.png')

'''fig, ax = plt.subplots(1, 4, figsize=(16, 4))

for i in range(4):
    X, Y = create_combined_trajectory(num_points_line=300, num_points_semicircle=200, num_points_convergence=100)
    ebm = train_ebm(X, Y, batch_size=128)    
    trajectory = generate_trajectory_ebm(ebm, np.array([0.0, 0.0]), 40)
    #ax[i].plot(np.linspace(0, 2*np.pi, 100), 2*np.sin(2*np.linspace(0, 2*np.pi, 100)), 'ro-', label='sin')
    # Plot the trajectory
    ax[i].plot(trajectory[:, 0], trajectory[:, 1], c='orange', label='Generated Trajectory')
    ax[i].legend()
    ax[i].grid(True)
    
plt.show()
plt.savefig('semicircle_random.png')
torch.save(ebm.state_dict(), 'ebm_semicircle_random.pth')'''