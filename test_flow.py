import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


# Simulate the trajectory (as defined earlier)
def simulate_trajectory(num_samples, amplitude=1.0, freq=1.0, random_sampling=False):
    if random_sampling:
        t_values = np.sort(np.random.uniform(0, 2 * np.pi, num_samples))
    else:
        t_values = np.linspace(0, 2 * np.pi, num_samples)

    x_values = t_values
    y_values = amplitude * np.sin(freq * t_values)
    trajectory = np.column_stack((x_values, y_values))

    # Apply random rotation
    rotation_matrix = np.random.randn(2, 2)
    q, _ = np.linalg.qr(rotation_matrix)
    rotated_trajectory = np.dot(trajectory, q)

    # Add continuous time variable to the trajectory
    final_trajectory = np.column_stack((t_values, rotated_trajectory))

    return final_trajectory


# Neural network to model f_theta(X)
class FlowModel(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=64):
        super(FlowModel, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, x):
        return self.network(x)


# Denoiser network to predict the noise
class DenoiserModel(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=64):
        super(DenoiserModel, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, x):
        return self.network(x)


# Interpolation function
def interpolate_position(time, time_start, time_end, pos_start, pos_end):
    weight = (time - time_start) / (time_end - time_start)
    position_interp = (1 - weight) * pos_start + weight * pos_end
    return position_interp


class FlowMatchingDataset(Dataset):
    def __init__(self, trajectory, denoising=False, noise_std=0.1):
        self.trajectory = trajectory
        self.denoising = denoising
        self.noise_std = noise_std

    def __len__(self):
        return len(self.trajectory) - 1

    def __getitem__(self, idx):
        t_start, x_start, y_start = self.trajectory[idx]
        t_end, x_end, y_end = self.trajectory[idx + 1]

        t_rand = np.random.uniform(t_start, t_end)
        x_interp = interpolate_position(t_rand, t_start, t_end, x_start, x_end)
        y_interp = interpolate_position(t_rand, t_start, t_end, y_start, y_end)

        dx_dt = (x_end - x_start) / (t_end - t_start)
        dy_dt = (y_end - y_start) / (t_end - t_start)

        clean_data = torch.tensor([x_interp, y_interp], dtype=torch.float32)
        target_data = torch.tensor([dx_dt, dy_dt], dtype=torch.float32)

        if self.denoising:
            noise = torch.randn_like(clean_data) * self.noise_std
            noisy_data = clean_data + noise
            return noisy_data, clean_data, target_data, -noise
        else:
            return clean_data, target_data


def train_flow_matching(
    trajectory,
    num_epochs=100,
    batch_size=32,
    learning_rate=0.001,
    denoising=False,
    noise_std=0.1,
):
    dataset = FlowMatchingDataset(trajectory, denoising=denoising, noise_std=noise_std)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = FlowModel()
    denoiser = DenoiserModel() if denoising else None
    criterion = nn.MSELoss()
    optimizer = optim.Adam(
        list(model.parameters()) + (list(denoiser.parameters()) if denoiser else []),
        lr=learning_rate,
    )

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for batch in dataloader:
            if denoising:
                noisy_data, clean_data, target_data, noise_target = batch
                optimizer.zero_grad()
                f_theta = model(noisy_data)
                denoise_pred = denoiser(noisy_data)
                loss = criterion(f_theta, target_data) + criterion(
                    denoise_pred, noise_target
                )
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            else:
                clean_data, target_data = batch
                optimizer.zero_grad()
                f_theta = model(clean_data)
                loss = criterion(f_theta, target_data)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

        if (epoch + 1) % 10 == 0:
            print(
                f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss / len(dataloader):.4f}"
            )

    return model, denoiser


# Global counter for function calls
call_count = 0


# Wrapper function to pass to solve_ivp that integrates using the neural network
def neural_ode(t, X, model):
    global call_count
    call_count += 1  # Increment the counter on each function call
    X_tensor = torch.tensor(X, dtype=torch.float32)
    dX_dt = model(X_tensor).detach().numpy()
    return dX_dt


def generate_trajectory(model, initial_point, t_values, denoiser=None, rtol=1e-2, atol=1e-4):
    global call_count
    call_count = 0  # Reset the counter at the beginning of each integration

    def combined_ode(t, X):
        global call_count
        call_count += 1
        X_tensor = torch.tensor(X, dtype=torch.float32)
        dx_dt = model(X_tensor).detach().numpy()
        if denoiser:
            denoise_correction = denoiser(X_tensor).detach().numpy()
            dx_dt += denoise_correction
        return dx_dt

    sol = solve_ivp(
        fun=combined_ode,
        t_span=(t_values[0], t_values[-1]),
        y0=initial_point,
        t_eval=t_values,
        method="RK23",
        rtol=rtol,
        atol=atol,
    )
    print(
        f"Number of function calls to neural_ode: {call_count}"
    )  # Print the number of function calls
    return sol.y.T  # Transpose to match the shape expected by the rest of the code


# Function to visualize the flow field along with the ground truth trajectory
def visualize_trajectory_with_flow_field(model, trajectory):
    t_values = trajectory[:, 0]
    x_values = trajectory[:, 1]
    y_values = trajectory[:, 2]

    plt.figure(figsize=(8, 6))
    plt.plot(x_values, y_values, label="Ground Truth Trajectory", color="blue")

    for t, x, y in zip(t_values, x_values, y_values):
        X_tensor = torch.tensor([x, y], dtype=torch.float32)
        flow = model(X_tensor).detach().numpy()
        plt.arrow(x, y, flow[0], flow[1], color="red", head_width=0.05, head_length=0.1)

    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Ground Truth Trajectory with Flow Field")
    plt.grid(True)
    plt.axis("equal")
    plt.legend()
    plt.show()


# Function to visualize the flow field around a given trajectory (either GT or BC)
def visualize_flow_field_around_trajectory(
    model,
    trajectory,
    grid_size=10,
    grid_extent=1.0,
    subsample_step=5,
    label="Trajectory",
    traj_color="blue",
    flow_color="red",
):
    t_values = trajectory[:, 0]
    x_values = trajectory[:, 1]
    y_values = trajectory[:, 2]

    plt.plot(x_values, y_values, label=f"{label}", color=traj_color)

    # Sub-sample the trajectory points
    for t, x, y in zip(
        t_values[::subsample_step],
        x_values[::subsample_step],
        y_values[::subsample_step],
    ):
        x_min, x_max = x - grid_extent, x + grid_extent
        y_min, y_max = y - grid_extent, y + grid_extent

        x_grid = np.linspace(x_min, x_max, grid_size)
        y_grid = np.linspace(y_min, y_max, grid_size)
        X_grid, Y_grid = np.meshgrid(x_grid, y_grid)

        for i in range(grid_size):
            for j in range(grid_size):
                X_tensor = torch.tensor(
                    [X_grid[i, j], Y_grid[i, j]], dtype=torch.float32
                )
                flow = model(X_tensor).detach().numpy()
                plt.arrow(
                    X_grid[i, j],
                    Y_grid[i, j],
                    flow[0] * 0.2,
                    flow[1] * 0.2,
                    color=flow_color,
                    head_width=0.03,
                    head_length=0.05,
                    alpha=0.7,
                )


# Function to plot and compare trajectories
def plot_trajectories(gt_trajectory, generated_trajectory):
    plt.figure(figsize=(8, 6))
    plt.plot(
        gt_trajectory[:, 1],
        gt_trajectory[:, 2],
        label="Ground Truth Trajectory",
        color="blue",
    )
    plt.plot(
        generated_trajectory[:, 0],
        generated_trajectory[:, 1],
        label="Generated Trajectory",
        color="red",
        linestyle="dashed",
    )
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Comparison of Ground Truth and Generated Trajectories")
    plt.legend()
    plt.grid(True)
    plt.axis("equal")
    plt.show()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Flow-based Behavior Cloning with Denoising"
    )

    # Add arguments for hyperparameters
    parser.add_argument(
        "--num_samples",
        type=int,
        default=100,
        help="Number of samples in the trajectory",
    )
    parser.add_argument(
        "--num_epochs", type=int, default=100, help="Number of epochs for training"
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size for training"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=0.001, help="Learning rate for training"
    )
    parser.add_argument(
        "--noise_std",
        type=float,
        default=0.1,
        help="Standard deviation of Gaussian noise added for denoising",
    )
    parser.add_argument(
        "--rtol", type=float, default=1e-1, help="Relative tolerance for solve_ivp"
    )
    parser.add_argument(
        "--atol", type=float, default=1e-2, help="Absolute tolerance for solve_ivp"
    )
    parser.add_argument('--no-denoising', action='store_false', dest='denoising_enabled', help='Disable denoising during training')

    args = parser.parse_args()
    return args


def main(args):
    # Simulate the trajectory
    gt_trajectory = simulate_trajectory(args.num_samples)

    # Train without denoising if --no-denoising is provided
    trained_model_no_denoiser, _ = train_flow_matching(
        gt_trajectory,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        denoising=not args.denoising_enabled,  # This should be `denoising_enabled`
    )

    # Train with denoising (default) unless --no-denoising is provided
    if args.denoising_enabled:
        trained_model_with_denoiser, denoiser = train_flow_matching(
            gt_trajectory,
            num_epochs=args.num_epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            denoising=True,
            noise_std=args.noise_std,
        )
    else:
        trained_model_with_denoiser, denoiser = None, None

    # Generate trajectories
    t_values = gt_trajectory[:, 0]
    initial_point = gt_trajectory[0, 1:]  # Starting point (x, y)

    # Without denoiser
    generated_trajectory_no_denoiser = generate_trajectory(
        trained_model_no_denoiser,
        initial_point,
        t_values,
        denoiser=None,
        rtol=args.rtol,
        atol=args.atol,
    )

    # With denoiser (if enabled)
    if args.denoising_enabled:
        generated_trajectory_with_denoiser = generate_trajectory(
            trained_model_with_denoiser,
            initial_point,
            t_values,
            denoiser=denoiser,
            rtol=args.rtol,
            atol=args.atol,
        )
        generated_trajectory_with_time_denoiser = np.column_stack(
            (t_values, generated_trajectory_with_denoiser)
        )
    else:
        generated_trajectory_with_time_denoiser = None

    # Add time to the generated trajectory without denoiser
    generated_trajectory_with_time_no_denoiser = np.column_stack(
        (t_values, generated_trajectory_no_denoiser)
    )

    # Create a single plot
    plt.figure(figsize=(10, 8))

    # Plot Ground Truth Trajectory
    visualize_flow_field_around_trajectory(
        trained_model_no_denoiser,
        gt_trajectory,
        grid_size=5,
        grid_extent=0.5,
        subsample_step=10,
        label="Ground Truth Trajectory",
        traj_color="blue",
        flow_color="lightblue",
    )

    # Plot Generated Trajectory without Denoiser
    visualize_flow_field_around_trajectory(
        trained_model_no_denoiser,
        generated_trajectory_with_time_no_denoiser,
        grid_size=5,
        grid_extent=0.5,
        subsample_step=10,
        label="Generated Trajectory (No Denoiser)",
        traj_color="red",
        flow_color="orange",
    )

    # Plot Generated Trajectory with Denoiser (if enabled)
    if args.denoising_enabled:
        visualize_flow_field_around_trajectory(
            trained_model_with_denoiser,
            generated_trajectory_with_time_denoiser,
            grid_size=5,
            grid_extent=0.5,
            subsample_step=10,
            label="Generated Trajectory (With Denoiser)",
            traj_color="green",
            flow_color="lightgreen",
        )

    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Comparison of Flow Fields and Trajectories")
    plt.legend()
    plt.grid(True)
    plt.axis("equal")
    plt.show()

if __name__ == "__main__":
    args = parse_args()
    main(args)