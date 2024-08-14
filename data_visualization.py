import zarr
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba

# Load dataset
dataset_path = './dataset/pusht_cchi_v7_replay.zarr.zip'  # Replace with the actual dataset path
path_size = -1  # Replace with the appropriate path_size value

dataset_root = zarr.open(dataset_path, 'r')
if path_size == -1:
    episode_ends = dataset_root['meta']['episode_ends'][:]

actions = dataset_root['data']['action'][:]
obs = dataset_root['data']['state'][:]

trajs = []
last_end_ind = 0
for end_ind in episode_ends:
    trajs.append({'obs': dataset_root['data']['state'][last_end_ind:end_ind],
                  'action': dataset_root['data']['action'][last_end_ind:end_ind]})
    last_end_ind = end_ind
    
print(f"dataset contains {len(episode_ends)} trajectories")
traj_lengths = [episode_ends[i + 1] - episode_ends[i] for i in range(len(episode_ends) - 1)]
traj_lengths.append(episode_ends[0])

# Create a histogram
plt.figure(figsize=(10, 6))
plt.hist(traj_lengths, bins=10, edgecolor='black')

# Adding title and labels
plt.title('Histogram of Time Steps per Trajectory')
plt.xlabel('Number of Time Steps')
plt.ylabel('Frequency')

# Display the histogram
plt.show()

trajectory_idx = 0  # Index of the trajectory to visualize
obs_seq, action_seq = trajs[trajectory_idx]['obs'], trajs[trajectory_idx]['action']

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_title(f'Trajectory {trajectory_idx}')
print(len(obs_seq))

# Set the color map for the alpha scale
cmap = plt.get_cmap('viridis')
num_steps = len(obs_seq)

# Use these lists to manage the labels
agent_label_set = False
block_label_set = False

for i in range(len(obs_seq)):
    obs = obs_seq[i]
    action = action_seq[i]
    agent_x, agent_y, block_x, block_y, block_angle = obs
    target_agent_x, target_agent_y = action

    # Compute the alpha value based on the time step
    alpha = (i + 1) / num_steps

    # Get the color with the computed alpha value
    color = to_rgba(cmap(alpha), alpha=alpha)

    print(f"Obs:        [agent_x,  agent_y,  block_x,  block_y,   block_angle]")
    print(f"Obs:       {repr(obs)}")
    print(f"Action:   [target_agent_x, target_agent_y] = ")
    print(f"Action:   {repr(action)}")

    # Set labels only once to avoid cluttering the legend
    agent_label = 'Agent' if not agent_label_set else None
    block_label = 'Block' if not block_label_set else None

    ax.scatter(agent_x, agent_y, 0, color=color, label=agent_label)
    ax.scatter(block_x, block_y, 0, color=color, label=block_label)

    agent_label_set = True
    block_label_set = True

# Adjust legend
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# Combine legends if necessary and ensure it's visible
handles, labels = ax.get_legend_handles_labels()
unique_labels = dict(zip(labels, handles))
ax.legend(unique_labels.values(), unique_labels.keys())

plt.show()