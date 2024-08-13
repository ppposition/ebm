import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from dataset.PushTdataset import PushTStateDataset
from environments.pushT import PushTEnv

# Load dataset
dataset_path = './dataset/pusht_cchi_v7_replay.zarr.zip'  # Replace with the actual dataset path
path_size = -1  # Replace with the appropriate path_size value
pred_horizon = 16
obs_horizon = 2
action_horizon = 8

dataset = PushTStateDataset(dataset_path, path_size, pred_horizon, obs_horizon, action_horizon)

# Visualize a single trajectory
env = PushTEnv()
env.seed(100000)

trajectory_idx = 0  # Index of the trajectory to visualize

print(f"dataset contains {len(dataset)} trajectorys")
traj_lengths = [len(dataset[i]['obs']) for i in range(len(dataset))]

# Create a histogram
plt.figure(figsize=(10, 6))
plt.hist(traj_lengths, bins=10, edgecolor='black')

# Adding title and labels
plt.title('Histogram of Time Steps per Trajectory')
plt.xlabel('Number of Time Steps')
plt.ylabel('Frequency')

# Display the histogram
plt.show()

obs_seq, action_seq = dataset[trajectory_idx]['obs'], dataset[trajectory_idx]['action']

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_title(f'Trajectory {trajectory_idx}')
print(len(obs_seq))

for i in range(len(obs_seq)):
    obs = obs_seq[i]
    action = action_seq[i]
    agent_x, agent_y, block_x, block_y, block_angle = obs
    target_agent_x, target_agent_y = action
    print(f"Obs:        [agent_x,  agent_y,  block_x,  block_y,   block_angle]")
    print(f"Obs:       {repr(obs)}")
    print(f"Action:   [target_agent_x, target_agent_y] = ")
    print(f"Action:   {repr(action)}")

    ax.scatter(agent_x, agent_y, 0, color='r', label='Agent' if i == 0 else None)
    ax.scatter(block_x, block_y, 0, color='b', label='Block' if i == 0 else None)
    ax.quiver(agent_x, agent_y, 0, target_agent_x - agent_x, target_agent_y - agent_y, 0, color='g', label='Action' if i == 0 else None)

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.legend()
plt.show()

# Load dataset statistics
stats_path = 'result/68/d1.pkl'  # Replace with the actual path to the statistics file
with open(stats_path, 'rb') as f:
    stats = pickle.load(f)

print("Dataset Statistics:")
print(stats)

# Normalize data
def normalize_data(data, stats):
    ndata = (data - stats['min']) / (stats['max'] - stats['min'])
    ndata = ndata * 2 - 1
    return ndata

# Example usage: Normalize a single observation
obs_example = obs_seq[0]
normalized_obs = normalize_data(obs_example, stats)
print("Normalized Observation Example:")
print(normalized_obs)
