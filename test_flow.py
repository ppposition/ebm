import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchsde
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


# Simulate the trajectory
def simulate_trajectory(num_samples, amplitude=1.0, freq=1.0, random_sampling=False, noise_std=0.0, sharp_turns=False):
    """
    Simulate a 2D trajectory, optionally with sharp turns and noise.

    Args:
        num_samples (int): Number of samples in the trajectory.
        amplitude (float): Amplitude of the trajectory.
        freq (float): Frequency of the trajectory (for sinusoidal curves).
        random_sampling (bool): Whether to use random sampling for time steps.
        noise_std (float): Standard deviation of Gaussian noise added to the trajectory.
        sharp_turns (bool): Whether to add sharp turns to the trajectory.

    Returns:
        np.ndarray: The trajectory with time indices.
    """
    if random_sampling:
        t_values = np.sort(np.random.uniform(0, 2 * np.pi, num_samples))
    else:
        t_values = np.linspace(0, 2 * np.pi, num_samples)

    if sharp_turns:
        # Sharp turns (e.g., square wave pattern)
        x_values = amplitude * np.sign(np.sin(freq * t_values))  # Creates sharp transitions
        y_values = amplitude * np.sign(np.cos(freq * t_values))
    else:
        # Smooth sinusoidal curve
        x_values = t_values
        y_values = amplitude * np.sin(freq * t_values)

    trajectory = np.column_stack((x_values, y_values))

    # Apply random rotation
    rotation_matrix = np.random.randn(2, 2)
    q, _ = np.linalg.qr(rotation_matrix)
    rotated_trajectory = np.dot(trajectory, q)

    # Add continuous time variable to the trajectory
    final_trajectory = np.column_stack((t_values, rotated_trajectory))

    # Add synthetic Gaussian noise if noise_std > 0
    if noise_std > 0:
        noise = np.random.normal(0, noise_std, final_trajectory[:, 1:].shape)
        final_trajectory[:, 1:] += noise

    return final_trajectory


# Neural network to model f_theta(X)
class MLP(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=64):
        super(MLP, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, x):
        return self.network(x)
    
class DiffusionMLP(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=64, min_val=-10.0, max_val=5.0):
        super(DiffusionMLP, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Tanh()  # Bound the output between -1 and 1
        )
        # max and min of the log of diffusion coefficient g
        self.min_val = min_val
        self.max_val = max_val

    def forward(self, x):
        # Apply the network, bound it, and then exponentiate to ensure positive output
        bounded_output = self.network(x)
        
        # Rescale from [-1, 1] to [min_val, max_val]
        scaled_output = (bounded_output + 1) * 0.5 * (self.max_val - self.min_val) + self.min_val
        
        # Optionally apply exponential if needed for larger dynamic range
        scaled_output = torch.exp(scaled_output)
        
        return scaled_output


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

        delta_t = t_end - t_start  # Dynamically computed delta_t
        assert delta_t >= 1e-4, "Delta t is too small"

        clean_data = torch.tensor([x_interp, y_interp], dtype=torch.float32)
        target_data = torch.tensor([dx_dt, dy_dt], dtype=torch.float32)
        delta_t_tensor = torch.tensor(
            delta_t, dtype=torch.float32
        )  # Return delta_t as a tensor

        if self.denoising:
            noise = torch.randn_like(clean_data) * self.noise_std
            noisy_data = clean_data + noise
            return noisy_data, clean_data, target_data, -noise, delta_t_tensor
        else:
            return clean_data, target_data, delta_t_tensor


# Update the negative log-likelihood function to accept delta_t dynamically
def negative_log_likelihood(f_pred, g_pred, dx_dt, delta_t):
    """
    Compute the negative log-likelihood loss for the SDE with a vector-valued diffusion term.

    Args:
        f_pred (torch.Tensor): The predicted drift values (f(x)).
        g_pred (torch.Tensor): The predicted diffusion values (g(x)), assumed to be vector-valued.
        dx_dt (torch.Tensor): The observed trajectory finite-difference values.
        delta_t (torch.Tensor): The time step size for each data point.

    Returns:
        torch.Tensor: The computed negative log-likelihood loss.
    """
    # Drift loss term: sum over both components
    drift_loss = torch.sum(
        ((dx_dt - f_pred) ** 2 * delta_t.unsqueeze(1)) / (2 * g_pred**2 + 1e-6), dim=1
    )

    # Normalization term: sum over both components
    norm_term = torch.sum(
        0.5 * torch.log(2 * torch.pi * g_pred**2 * delta_t.unsqueeze(1) + 1e-6), dim=1
    )

    # Final NLL: sum over all data points
    nll = torch.sum(drift_loss + norm_term)

    return nll


def train_flow_matching(
    trajectory,
    num_epochs=100,
    batch_size=32,
    learning_rate=0.001,
    denoising=False,
    noise_std=0.1,
    l2_reg=0.0,
    diffusion_min=-10.0,
    diffusion_max=5.0,  # Add these to control the diffusion range
):
    dataset = FlowMatchingDataset(trajectory, denoising=denoising, noise_std=noise_std)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    flow_field = MLP()  # For drift (f_theta)
    denoiser = MLP() if denoising else None
    diffusion_field = DiffusionMLP(min_val=diffusion_min, max_val=diffusion_max)  # For diffusion (g_theta)

    optimizer = optim.Adam(
        list(flow_field.parameters())
        + (list(denoiser.parameters()) if denoiser else [])
        + list(diffusion_field.parameters()),
        lr=learning_rate,
        weight_decay=l2_reg,
    )

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for batch in dataloader:
            optimizer.zero_grad()

            if denoising:
                noisy_data, clean_data, target_data, noise_target, delta_t = batch
            else:
                clean_data, target_data, delta_t = batch

            # Predict drift (f_theta)
            f_theta = flow_field(clean_data if not denoising else noisy_data)
            # Predict diffusion (g_theta) with bounded positive output
            g_theta = diffusion_field(clean_data if not denoising else noisy_data)

            # Compute MLE-based negative log-likelihood loss
            loss = negative_log_likelihood(f_theta, g_theta, target_data, delta_t)

            # If denoising, add the denoising loss
            if denoising:
                denoise_pred = denoiser(noisy_data)
                denoise_loss = nn.MSELoss()(denoise_pred, noise_target)
                loss += denoise_loss

            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        if (epoch + 1) % 10 == 0:
            print(
                f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss / len(dataloader):.4f}"
            )

    return flow_field, denoiser, diffusion_field


class FlowSDE(torchsde.SDEIto):  # Assuming Ito SDEs are supported by default

    def __init__(self, model, diffusion_model, denoiser=None):
        super().__init__(noise_type="diagonal")  # Removed `sde_type`
        self.model = model  # The drift model
        self.diffusion_model = diffusion_model  # The learned diffusion model
        self.denoiser = denoiser

    # Drift term: f + denoising
    def f(self, t, X):
        with torch.no_grad():
            flow = self.model(X)
            if self.denoiser:
                denoising = self.denoiser(X)
                flow += denoising
            return flow

    # Diffusion term: g(X)
    def g(self, t, X):
        with torch.no_grad():
            diffusion = self.diffusion_model(X)
            print(diffusion)
            return diffusion  # Return the learned diffusion coefficients


def generate_trajectory(
    model,
    diffusion_model,  # Added diffusion model as a parameter
    initial_point,
    t_values,
    denoiser=None,
    rtol=1e-1,
    atol=1e-2,
    use_sde=False,
):
    global call_count
    call_count = 0  # Reset the counter at the beginning of each integration

    # ODE version: Use combined_ode with call counting
    def combined_ode(t, X):
        global call_count
        call_count += 1
        X_tensor = torch.tensor(X, dtype=torch.float32)
        dx_dt = model(X_tensor).detach().numpy()
        if denoiser:
            denoise_correction = denoiser(X_tensor).detach().numpy()
            dx_dt += denoise_correction
        return dx_dt

    # SDE version: Wrap the model and diffusion model to count function calls
    class CountingFlowSDE(FlowSDE):
        def f(self, t, X):
            global call_count
            call_count += 1
            return super().f(t, X)

    if use_sde:
        # Use SDE solver with call counting
        sde_model = CountingFlowSDE(model, diffusion_model, denoiser=denoiser)
        initial_point_tensor = torch.tensor(
            initial_point, dtype=torch.float32
        ).unsqueeze(0)
        ts = torch.tensor(t_values, dtype=torch.float32)
        with torch.no_grad():
            trajectory = (
                torchsde.sdeint(
                    sde_model,
                    initial_point_tensor,
                    ts,
                    method="milstein",  # 'srk', 'euler', 'milstein', etc.
                    adaptive=True,
                    rtol=rtol,
                    atol=atol,
                )
                .squeeze(1)
                .numpy()
            )  # Squeeze to remove batch dimension
    else:
        # Use ODE solver with call counting
        sol = solve_ivp(
            fun=combined_ode,
            t_span=(t_values[0], t_values[-1]),
            y0=initial_point,
            t_eval=t_values,
            method="RK23",
            rtol=rtol,
            atol=atol,
        )
        trajectory = sol.y.T

    print(f"Number of function calls: {call_count}")
    return trajectory


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


def visualize_flow_field_around_trajectory(
    model,
    trajectory,
    denoiser=None,  # Add the denoiser as an optional argument
    grid_size=5,
    grid_extent=0.5,
    subsample_step=10,
    label="Trajectory",
    traj_color="blue",
    flow_color="red",
):
    t_values = trajectory[:, 0]
    x_values = trajectory[:, 1]
    y_values = trajectory[:, 2]

    plt.plot(x_values, y_values, label=label, color=traj_color)

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
                
                # Calculate f(X)
                flow = model(X_tensor).detach().numpy()
                
                # Add the denoising direction if denoiser is provided
                if denoiser:
                    denoise_correction = denoiser(X_tensor).detach().numpy()
                    flow += denoise_correction  # Composite direction
                
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
    parser = argparse.ArgumentParser(description="Flow-based Behavior Cloning with Denoising and Noisy GT Data")

    # Add arguments for hyperparameters
    parser.add_argument(
        "--num_samples",
        type=int,
        default=30,
        help="Number of samples in the trajectory",
    )
    parser.add_argument(
        "--num_epochs", type=int, default=1000, help="Number of epochs for training"
    )
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for training')
    parser.add_argument(
        "--noise_std",
        type=float,
        default=0.2,
        help="Standard deviation of Gaussian noise added for denoising",
    )
    parser.add_argument(
        "--gt_noise_std",
        type=float,
        default=0.01,
        help="Standard deviation of Gaussian noise added to the ground truth trajectory",
    )
    parser.add_argument('--rtol', type=float, default=1e-2, help='Relative tolerance for solve_ivp')
    parser.add_argument('--atol', type=float, default=1e-4, help='Absolute tolerance for solve_ivp')
    parser.add_argument('--no-denoising', action='store_false', dest='denoising_enabled', help='Disable denoising during training')
    parser.add_argument('--sharp_turns', action='store_true', help='Add sharp turns to the trajectory')
    parser.add_argument(
        "--l2_reg",
        type=float,
        default=0.01,
        help="L2 regularization strength (weight decay)",
    )
    parser.add_argument(
        "--use_sde", action="store_true", help="Use SDE instead of ODE for inference"
    )

    args = parser.parse_args()
    return args


def main(args):
    # Simulate the trajectory with optional sharp turns and noise
    gt_trajectory = simulate_trajectory(
        args.num_samples,
        noise_std=args.gt_noise_std,  # Pass the ground truth noise standard deviation
        sharp_turns=args.sharp_turns  # Pass the sharp turns flag
    )

    # Train without denoising if --no-denoising is provided
    trained_model_no_denoiser, _, diffusion_model_no_denoiser = train_flow_matching(
        gt_trajectory,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        denoising=False,
        l2_reg=args.l2_reg,  # Pass the L2 regularization strength
    )

    # Train with denoising (default) unless --no-denoising is provided
    if args.denoising_enabled:
        trained_model_with_denoiser, denoiser, diffusion_model_with_denoiser = (
            train_flow_matching(
                gt_trajectory,
                num_epochs=args.num_epochs,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                denoising=True,
                noise_std=args.noise_std,
                l2_reg=args.l2_reg,  # Pass the L2 regularization strength
            )
        )
    else:
        trained_model_with_denoiser, denoiser, diffusion_model_with_denoiser = (
            None,
            None,
            None,
        )

    # Generate trajectories
    t_values = gt_trajectory[:, 0]
    initial_point = gt_trajectory[0, 1:]  # Starting point (x, y)

    # Without denoiser
    generated_trajectory_no_denoiser = generate_trajectory(
        trained_model_no_denoiser,
        diffusion_model_no_denoiser,  # Add diffusion model
        initial_point,
        t_values,
        denoiser=None,
        rtol=args.rtol,
        atol=args.atol,
        use_sde=args.use_sde,
    )

    # With denoiser (if enabled)
    if args.denoising_enabled:
        generated_trajectory_with_denoiser = generate_trajectory(
            trained_model_with_denoiser,
            diffusion_model_with_denoiser,  # Add diffusion model
            initial_point,
            t_values,
            denoiser=denoiser,
            rtol=args.rtol,
            atol=args.atol,
            use_sde=args.use_sde,
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
        subsample_step=5,
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
        subsample_step=5,
        label="Generated Trajectory (No Denoiser)",
        traj_color="red",
        flow_color="orange",
    )

    # Plot Generated Trajectory with Denoiser (if enabled)
    if args.denoising_enabled:
        visualize_flow_field_around_trajectory(
            trained_model_with_denoiser,
            generated_trajectory_with_time_denoiser,
            denoiser=denoiser,
            grid_size=5,
            grid_extent=0.5,
            subsample_step=5,
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
