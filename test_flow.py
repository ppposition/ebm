import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchsde
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.integrate import solve_ivp


# Simulate the trajectory
def simulate_trajectory(
    num_trajectories,
    num_samples,
    amplitude=1.0,
    freq=1.0,
    random_sampling=False,
    noise_std=0.0,
    sharp_turns=False,
    branching=False
):
    """
    Simulate one or more 2D trajectories, optionally with branching and noise.

    Args:
        num_trajectories (int): Number of trajectories.
        num_samples (int): Number of samples in each trajectory.
        amplitude (float): Amplitude of the trajectory.
        freq (float): Frequency of the trajectory (for sinusoidal curves).
        random_sampling (bool): Whether to use random sampling for time steps.
        noise_std (float): Standard deviation of Gaussian noise added to the trajectory.
        sharp_turns (bool): Whether to add sharp turns to the trajectory.
        branching (bool): Whether to branch the trajectory into two directions.

    Returns:
        list of np.ndarray: List of trajectories with time indices.
    """
    trajectories = []

    for _ in range(num_trajectories):
        if random_sampling:
            t_values = np.sort(np.random.uniform(0, 2 * np.pi, num_samples))
        else:
            t_values = np.linspace(0, 2 * np.pi, num_samples)

        if branching:
            # Create a smooth line that branches into two directions at the halfway point
            x_values = t_values
            y_values = np.zeros_like(t_values)

            # Create branching
            branch_point = num_samples // 2

            # Line before branching (y = 0)
            y_values[:branch_point] = 0

            # Create two separate branches after the branch point
            # Using a smooth transition instead of linear interpolation
            smooth_transition = np.sin(np.linspace(0, np.pi, num_samples - branch_point))
            y_branch_1 = amplitude * smooth_transition  # First branch
            y_branch_2 = -amplitude * smooth_transition  # Second branch

            # Construct the trajectories
            trajectory_1 = np.column_stack((t_values[:branch_point], x_values[:branch_point], y_values[:branch_point]))
            trajectory_2_1 = np.column_stack((t_values[branch_point:], x_values[branch_point:], y_branch_1))
            trajectory_2_2 = np.column_stack((t_values[branch_point:], x_values[branch_point:], y_branch_2))

            # Append both branches to the list of trajectories
            trajectories.append(np.vstack((trajectory_1, trajectory_2_1)))
            trajectories.append(np.vstack((trajectory_1, trajectory_2_2)))
        else:
            # Simple straight line trajectory
            x_values = t_values
            y_values = amplitude * np.sin(freq * t_values)
            trajectory = np.column_stack((t_values, x_values, y_values))
            trajectories.append(trajectory)

        # Add synthetic Gaussian noise if noise_std > 0
        if noise_std > 0:
            for trajectory in trajectories:
                noise = np.random.normal(0, noise_std, trajectory[:, 1:].shape)
                trajectory[:, 1:] += noise

    return trajectories


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
    def __init__(self, trajectories, denoising=False, noise_std=0.1):
        """
        Args:
            trajectories (list of np.ndarray): List of 2D trajectories, each with shape (num_samples, 3).
            denoising (bool): Whether to apply denoising to the input data.
            noise_std (float): Standard deviation of noise for denoising.
        """
        self.trajectories = trajectories  # List of trajectories
        self.denoising = denoising
        self.noise_std = noise_std
        self.data = []  # Flattened list of all trajectory segments

        # Flatten the trajectories into a single dataset
        for trajectory in trajectories:
            for idx in range(len(trajectory) - 1):
                self.data.append((trajectory[idx], trajectory[idx + 1]))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        start_point, end_point = self.data[idx]  # Correct unpacking
        t_start, x_start, y_start = start_point
        t_end, x_end, y_end = end_point

        t_rand = np.random.uniform(t_start, t_end)
        x_interp = interpolate_position(t_rand, t_start, t_end, x_start, x_end)
        y_interp = interpolate_position(t_rand, t_start, t_end, y_start, y_end)

        dx_dt = (x_end - x_start) / (t_end - t_start)
        dy_dt = (y_end - y_start) / (t_end - t_start)

        delta_t = t_end - t_start  # Dynamically computed delta_t
        assert delta_t >= 1e-4, f"Delta t is too small: {delta_t}"

        clean_data = torch.tensor([x_interp, y_interp], dtype=torch.float32)
        target_data = torch.tensor([dx_dt, dy_dt], dtype=torch.float32)
        delta_t_tensor = torch.tensor(delta_t, dtype=torch.float32)  # Return delta_t as a tensor

        if self.denoising:
            noise = torch.randn_like(clean_data) * self.noise_std
            noisy_data = clean_data + noise
            return noisy_data, clean_data, target_data, -noise, delta_t_tensor
        else:
            return clean_data, target_data, delta_t_tensor


# Updated negative log-likelihood function for the improved training strategy
def improved_negative_log_likelihood(f_pred, dx_dt):
    """
    Compute the log-squared error objective for training the flow network.

    This loss function is derived from the original negative log-likelihood (NLL) of a
    Stochastic Differential Equation (SDE) model, where the drift function (f_pred)
    and the diffusion function (g_pred) are trained together.

    Derivation:
    1. The original NLL can be expressed as:
       NLL = sum(((dx/dt - f_pred) ** 2 / (2 * g_pred ** 2) + 0.5 * log(2 * pi * g_pred ** 2)) * delta_t)

    2. By analytically minimizing the NLL with respect to g_pred, we find that the
       optimal g_pred is given by:
       g_pred ** 2 = (dx/dt - f_pred) ** 2 * delta_t

    3. Substituting this optimal g_pred back into the NLL and removing constant terms,
       the resulting objective simplifies to:
       J(f_pred) = log((dx/dt - f_pred) ** 2)

    This new objective focuses solely on minimizing the difference between the observed
    trajectory (dx/dt) and the predicted drift (f_pred), without explicitly training the
    diffusion term. The log term helps in reducing the impact of large errors, promoting
    smoother convergence.

    Args:
        f_pred (torch.Tensor): The predicted drift values (f(x)).
        dx_dt (torch.Tensor): The observed trajectory finite-difference values.

    Returns:
        torch.Tensor: The computed log-squared error loss for training the flow network.
    """
    log_sq_error = torch.log(torch.sum(torch.square(dx_dt - f_pred), dim=1) + 1e-6)
    return torch.sum(log_sq_error)


def diffusion_regression_loss(g_pred, f_pred, dx_dt, delta_t):
    """
    Compute the regression loss for the diffusion network.
    g_pred^2 should match (dx/dt - f)^2 * delta_t
    """
    # Detach f_pred to prevent backpropagation through f_theta
    target_g_sq = (dx_dt - f_pred.detach()) ** 2 * delta_t.unsqueeze(1)
    g_sq = g_pred**2
    return torch.sum(torch.square(g_sq - target_g_sq))


def train_flow_matching(
    trajectories,
    num_epochs=100,
    batch_size=32,
    learning_rate=0.001,
    denoising=False,
    noise_std=0.1,
    l2_reg=0.0,
    diffusion_min=-10.0,
    diffusion_max=0.0,
    initial_epochs_for_flow_only=100,  # Number of epochs to train flow only
):
    dataset = FlowMatchingDataset(trajectories, denoising=denoising, noise_std=noise_std)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    flow_field = MLP()  # For drift (f_theta)
    denoiser = MLP() if denoising else None
    diffusion_field = DiffusionMLP(min_val=diffusion_min, max_val=diffusion_max)  # For diffusion (g_theta)

    # Separate optimizers for flow, denoising, and diffusion
    optimizer_flow = optim.Adam(
        flow_field.parameters(),
        lr=learning_rate,
        weight_decay=l2_reg,
    )
    
    optimizer_denoiser = optim.Adam(
        denoiser.parameters(),
        lr=learning_rate,
        weight_decay=l2_reg,
    ) if denoiser else None
    
    optimizer_diffusion = optim.Adam(
        diffusion_field.parameters(),
        lr=learning_rate,
        weight_decay=l2_reg,
    )

    for epoch in range(num_epochs):
        epoch_loss_flow = 0.0
        epoch_loss_diffusion = 0.0
        epoch_loss_denoising = 0.0

        for batch in dataloader:
            optimizer_flow.zero_grad()
            if epoch >= initial_epochs_for_flow_only:
                optimizer_diffusion.zero_grad()
            if optimizer_denoiser:
                optimizer_denoiser.zero_grad()

            if denoising:
                noisy_data, clean_data, target_data, noise_target, delta_t = batch
            else:
                clean_data, target_data, delta_t = batch

            # Predict drift (f_theta)
            f_theta = flow_field(clean_data if not denoising else noisy_data)

            if epoch >= initial_epochs_for_flow_only:
                # Predict diffusion (g_theta) with bounded positive output, after initial flow training
                g_theta = diffusion_field(clean_data if not denoising else noisy_data)
                # Compute regression loss for diffusion
                diffusion_loss = diffusion_regression_loss(g_theta, f_theta, target_data, delta_t)
                diffusion_loss.backward()
                epoch_loss_diffusion += diffusion_loss.item()
                optimizer_diffusion.step()

            # Compute improved negative log-likelihood loss for flow
            flow_loss = improved_negative_log_likelihood(f_theta, target_data)
            flow_loss.backward()
            epoch_loss_flow += flow_loss.item()
            optimizer_flow.step()

            # Compute denoising loss if applicable
            if denoising:
                denoise_pred = denoiser(noisy_data)
                denoise_loss = nn.MSELoss()(denoise_pred, noise_target)
                denoise_loss.backward()
                epoch_loss_denoising += denoise_loss.item()
                optimizer_denoiser.step()  # Separate step for denoiser

        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch + 1}/{num_epochs}], "
                  f"Flow Loss: {epoch_loss_flow / len(dataloader):.4f}, "
                  f"Diffusion Loss: {epoch_loss_diffusion / len(dataloader):.4f}, "
                  f"Denoising Loss: {epoch_loss_denoising / len(dataloader):.4f}")

    return flow_field, denoiser, diffusion_field


class FlowSDE(torchsde.SDEIto):  # Assuming Ito SDEs are supported by default

    def __init__(
        self,
        model,
        diffusion_model,
        denoiser=None,
        denoising_magnitude=1.0,
        noise_std=0.1,
    ):
        super().__init__(noise_type="diagonal")
        self.model = model  # The drift model
        self.diffusion_model = diffusion_model  # The learned diffusion model
        self.denoiser = denoiser
        self.denoising_magnitude = denoising_magnitude  # Scaling factor for denoising
        self.noise_std = noise_std

    # Drift term: f + c * denoising
    def f(self, t, X):
        with torch.no_grad():
            flow = self.model(X)
            if self.denoiser:
                # derivation of the denoising_coefficient involves using fokker-planck equation
                # and Ito's lemma to derive the evolution of logP, and enforcing that the
                # combined diffusion term is negative (probablility concentrate)
                # we need to enforce that self.denoising_magnitude >= 1.0
                denoising_coefficient = (
                    0.5
                    * torch.sum(self.g(t, X))
                    * self.denoising_magnitude
                    * self.noise_std**2
                )
                denoising = self.denoiser(X) * denoising_coefficient
                flow += denoising
            return flow

    # Diffusion term: g(X)
    def g(self, t, X):
        with torch.no_grad():
            diffusion = self.diffusion_model(X)
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
    denoising_magnitude=1.0,  # Add denoising magnitude
    noise_std=0.1,
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
            denoise_correction = denoiser(X_tensor).detach().numpy() * denoising_magnitude
            dx_dt += denoise_correction
        return dx_dt

    # SDE version: Wrap the model and diffusion model to count function calls
    class CountingFlowSDE(FlowSDE):

        def __init__(
            self,
            model,
            diffusion_model,
            denoiser=None,
            denoising_magnitude=1.0,
            noise_std=0.1,
        ):
            super().__init__(
                model, diffusion_model, denoiser, denoising_magnitude, noise_std
            )

        def f(self, t, X):
            global call_count
            call_count += 1
            return super().f(t, X)

    if use_sde:
        # Use SDE solver with call counting
        sde_model = CountingFlowSDE(
            model,
            diffusion_model,
            denoiser=denoiser,
            denoising_magnitude=denoising_magnitude,
            noise_std=noise_std,
        )
        initial_point_tensor = torch.tensor(initial_point, dtype=torch.float32).unsqueeze(0)
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


def visualize_flow_field_around_trajectory(
    model,
    trajectory,
    diffusion_model=None,
    denoiser=None,  # Add the denoiser as an optional argument
    grid_size=5,
    grid_extent=0.5,
    subsample_step=10,
    label="Trajectory",
    traj_color="blue",
    flow_color="red",
    denoising_magnitude=1.0,
    plot_ellipsoids=True,
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

        # Plot ellipsoid if enabled and diffusion model is provided
        if plot_ellipsoids and diffusion_model is not None:
            X_tensor = torch.tensor([x, y], dtype=torch.float32)
            diffusion_std = diffusion_model(X_tensor).detach().numpy()

            # Create an ellipse for each point with width and height proportional to the std deviation
            ellipse = patches.Ellipse(
                (x, y),
                width=2 * diffusion_std[0],  # 2 * std for width
                height=2 * diffusion_std[1],  # 2 * std for height
                edgecolor="purple",
                facecolor="none",
                linestyle="--",
            )
            plt.gca().add_patch(ellipse)

        for i in range(grid_size):
            for j in range(grid_size):
                X_tensor = torch.tensor(
                    [X_grid[i, j], Y_grid[i, j]], dtype=torch.float32
                )

                # Calculate f(X)
                flow = model(X_tensor).detach().numpy()

                # Add the denoising direction if denoiser is provided
                if denoiser:
                    denoise_correction = denoising_magnitude * denoiser(X_tensor).detach().numpy()
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
        default=40,
        help="Number of samples in the trajectory",
    )
    parser.add_argument(
        "--num_epochs", type=int, default=4000, help="Number of epochs for training"
    )
    parser.add_argument(
        "--batch_size", type=int, default=256, help="Batch size for training"
    )
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for training')
    parser.add_argument(
        "--noise_std",
        type=float,
        default=0.05,
        help="Standard deviation of Gaussian noise added for denoising",
    )
    parser.add_argument(
        "--gt_noise_std",
        type=float,
        default=0.025,
        help="Standard deviation of Gaussian noise added to the ground truth trajectory",
    )
    parser.add_argument('--rtol', type=float, default=1e-2, help='Relative tolerance for solve_ivp')
    parser.add_argument('--atol', type=float, default=1e-4, help='Absolute tolerance for solve_ivp')
    parser.add_argument('--no-denoising', action='store_false', dest='denoising_enabled', help='Disable denoising during training')
    parser.add_argument('--sharp_turns', action='store_true', help='Add sharp turns to the trajectory')
    parser.add_argument(
        "--l2_reg",
        type=float,
        default=0.001,
        help="L2 regularization strength (weight decay)",
    )
    parser.add_argument(
        "--use_sde", action="store_true", help="Use SDE instead of ODE for inference"
    )
    parser.add_argument('--branching', action='store_true', help='Enable branching in the trajectory')
    parser.add_argument(
        '--denoising_magnitude',
        type=float,
        default=2.0,
        help='Scale factor for the denoising correction during inference'
    )
    args = parser.parse_args()
    return args


def main(args):
    if args.denoising_magnitude < 1.0 and args.use_sde:
        print(
            "Warning: denoising magnitude should be larger than 1.0 for SDE mode to compensate the diffusion term"
        )

    # Simulate the trajectory with optional sharp turns and noise
    gt_trajectories = simulate_trajectory(
        2,
        args.num_samples,
        noise_std=args.gt_noise_std,
        sharp_turns=args.sharp_turns,
        branching=args.branching,  # Pass the branching flag
    )

    # Train without denoising if --no-denoising is provided
    trained_model_no_denoiser, _, diffusion_model_no_denoiser = train_flow_matching(
        gt_trajectories,
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
                gt_trajectories,
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
    t_values = gt_trajectories[0][:, 0]
    initial_point = gt_trajectories[0][0, 1:]  # Starting point (x, y)

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
        denoising_magnitude=args.denoising_magnitude,
        noise_std=args.noise_std,
    )

    # Ensure the trajectory has the time dimension (t, x, y)
    generated_trajectory_no_denoiser = np.column_stack((t_values, generated_trajectory_no_denoiser))

    # Create a single plot
    plt.figure(figsize=(10, 8))

    # Plot Ground Truth Trajectory
    for traj in gt_trajectories:
        visualize_flow_field_around_trajectory(
            trained_model_no_denoiser,
            traj,  # Use the first trajectory for visualization
            diffusion_model=diffusion_model_no_denoiser,
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
        generated_trajectory_no_denoiser,
        diffusion_model=diffusion_model_no_denoiser,
        grid_size=5,
        grid_extent=0.5,
        subsample_step=5,
        label="Generated Trajectory (No Denoiser)",
        traj_color="red",
        flow_color="orange",
    )

    # Plot multiple Generated Trajectories with Denoiser (if enabled)
    if args.denoising_enabled:
        for i in range(20):  # Generate and plot 10 trajectories
            generated_trajectory_with_denoiser = generate_trajectory(
                trained_model_with_denoiser,
                diffusion_model_with_denoiser,  # Add diffusion model
                initial_point,
                t_values,
                denoiser=denoiser,
                rtol=args.rtol,
                atol=args.atol,
                use_sde=args.use_sde,
                denoising_magnitude=args.denoising_magnitude,
                noise_std=args.noise_std,
            )
            # Concatenate the time values to create (t, x, y) format
            generated_trajectory_with_denoiser = np.column_stack(
                (t_values, generated_trajectory_with_denoiser)
            )
            visualize_flow_field_around_trajectory(
                trained_model_with_denoiser,
                generated_trajectory_with_denoiser,
                diffusion_model=diffusion_model_with_denoiser,
                denoiser=denoiser,
                grid_size=5,
                grid_extent=0.5,
                subsample_step=5,
                label=f"Generated Trajectory {i + 1} (With Denoiser)",
                traj_color="green",
                flow_color="lightgreen",
                denoising_magnitude=args.denoising_magnitude,
            )

    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Comparison of Flow Fields and Trajectories")
    plt.legend(loc="best")
    plt.grid(True)
    plt.axis("equal")
    plt.show()

if __name__ == "__main__":
    args = parse_args()
    main(args)
