import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from vector import generate_trajectory, plot_vector_field_and_trajectories

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
N_TRAJECTORIES = 30
NUM_STEPS = 100
STEP_SIZE = 0.1
EPOCHS = 1000
BATCH_SIZE = 64
LEARNING_RATE = 0.001
HIDDEN_SIZES = [128, 512, 1024, 512, 128]

def generate_multiple_trajectories(n, num_steps, step_size):
    return [generate_trajectory(np.random.uniform(5, 10), np.random.uniform(-3, 3), 
                                steps=num_steps, step_size=step_size) for _ in range(n)]

def generate_dataset(trajectories):
    X, Y = [], []
    for trajectory in trajectories:
        X.extend(trajectory[:-1])
        Y.extend(trajectory[1:])
    return np.array(X), np.array(Y)

class ResBlock(nn.Module):
    def __init__(self, in_features, out_features):
        super(ResBlock, self).__init__()
        self.linear1 = nn.Linear(in_features, out_features)
        self.linear2 = nn.Linear(out_features, out_features)
        self.relu = nn.ReLU()
        
        if in_features != out_features:
            self.shortcut = nn.Linear(in_features, out_features)
        else:
            self.shortcut = nn.Identity()
    
    def forward(self, x):
        residual = x
        out = self.relu(self.linear1(x))
        out = self.linear2(out)
        out += self.shortcut(residual)
        return self.relu(out)

class TrajectoryMLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(TrajectoryMLP, self).__init__()
        layers = []
        prev_size = input_size
        for hidden_size in hidden_sizes:
            layers.append(ResBlock(prev_size, hidden_size))
            prev_size = hidden_size
        layers.append(nn.Linear(prev_size, output_size))
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)

def train_mlp(X, Y, hidden_sizes, epochs, batch_size, learning_rate):
    model = TrajectoryMLP(2, hidden_sizes, 2).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    
    dataset = torch.utils.data.TensorDataset(torch.tensor(X, dtype=torch.float32).to(device),
                                             torch.tensor(Y, dtype=torch.float32).to(device))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    for epoch in range(epochs):
        total_loss = 0
        for batch_X, batch_Y in dataloader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_Y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        if (epoch+1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Average Loss: {total_loss / len(dataloader):.4f}")
    
    return model

def generate_trajectory_mlp(model, start_point, num_steps):
    trajectory = [start_point]
    current_point = torch.tensor(start_point, dtype=torch.float32).unsqueeze(0).to(device)
    
    for _ in range(num_steps):
        next_point = model(current_point)
        trajectory.append(next_point.squeeze().cpu().detach().numpy())
        current_point = next_point
    
    return np.array(trajectory)

def plot_arrow_map(ax, model, x_range, y_range, density=20, arrow_len=1):
    x = np.linspace(x_range[0], x_range[1], density)
    y = np.linspace(y_range[0], y_range[1], density)
    X, Y = np.meshgrid(x, y)
    
    U, V = np.zeros_like(X), np.zeros_like(Y)
    
    for i in range(density):
        for j in range(density):
            input_tensor = torch.tensor([[X[i, j], Y[i, j]]], dtype=torch.float32).to(device)
            output = model(input_tensor).squeeze().cpu().detach().numpy()
            U[i, j], V[i, j] = (output - [X[i, j], Y[i, j]]) * arrow_len
    
    ax.quiver(X, Y, U, V, scale=30, width=0.002)

def main():
    fig, axs = plt.subplots(4, 2, figsize=(20, 32))

    for i in range(4):
        trajectories = generate_multiple_trajectories(N_TRAJECTORIES, NUM_STEPS, STEP_SIZE)
        X, Y = generate_dataset(trajectories)
        mlp = train_mlp(X, Y, HIDDEN_SIZES, EPOCHS, BATCH_SIZE, LEARNING_RATE)

        sample_points = np.random.uniform(low=[5, -2], high=[10, 2], size=(4, 2))

        for trajectory in trajectories:
            axs[i, 0].plot(trajectory[:, 0], trajectory[:, 1], alpha=0.5)
        plot_vector_field_and_trajectories(axs[i, 0], step_size=STEP_SIZE)
        axs[i, 0].set_title(f'Vector Field and Trajectories {i+1}')

        plot_arrow_map(axs[i, 1], mlp, x_range=(-10, 10), y_range=(-5, 5), arrow_len=1/STEP_SIZE)
        for j, point in enumerate(sample_points):
            mlp_trajectory = generate_trajectory_mlp(mlp, point, 100)
            axs[i, 1].plot(mlp_trajectory[:, 0], mlp_trajectory[:, 1], label=f'MLP Trajectory {j}')
            
            vector_trajectory = generate_trajectory(point[0], point[1], step_size=STEP_SIZE, steps=100)
            axs[i, 1].plot(vector_trajectory[:, 0], vector_trajectory[:, 1], c='blue', label=f'Vector Field Trajectory {j}')

        axs[i, 1].set_title(f'MLP and Vector Field Trajectories with Arrow Map {i+1}')
        axs[i, 1].legend()
        axs[i, 0].grid(True)
        axs[i, 1].grid(True)

    plt.tight_layout()
    plt.savefig('predict_vector_comparison.png')
    plt.close()

if __name__ == "__main__":
    main()