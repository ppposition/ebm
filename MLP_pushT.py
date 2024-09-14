import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import collections
from tqdm import tqdm
import zarr
from skvideo.io import vwrite

from dataset.PushTdataset import PushTStateDataset, normalize_data, unnormalize_data, get_data_stats
from environments.pushT import PushTEnv

class PushTdataset(torch.utils.data.Dataset):
    def __init__(self, dataset_path):
        dataset_root = zarr.open(dataset_path, 'r')
        train_data = {
            'action':dataset_root['data']['action'][:]-dataset_root['data']['state'][:][..., :2],
            'obs': dataset_root['data']['state'][:],
        }
        stats = dict()
        normalized_train_data = dict()
        for key, data in train_data.items():
            stats[key] = get_data_stats(data)
            normalized_train_data[key] = normalize_data(data, stats[key])
        self.stats = stats
        self.normalized_train_data = normalized_train_data
        self.episode_ends = set(dataset_root['meta']['episode_ends'][:])
        self.episode_ends.add(len(self.normalized_train_data['action'])-1)
        
    def __len__(self):
        return len(self.normalized_train_data['action'])

    def __getitem__(self, idx):
       nsample = dict()
       nsample['obs'] = self.normalized_train_data['obs'][idx]
       nsample['action'] = self.normalized_train_data['action'][idx]
       return nsample
   
# Hyperparameters
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DATASET_PATH = "dataset/pusht_cchi_v7_replay.zarr.zip"
PRED_HORIZON = 1
OBS_HORIZON = 1
ACTION_HORIZON = 1
INPUT_DIM = 5
OUTPUT_DIM = 2
HIDDEN_DIM = 512
HIDDEN_DEPTH = 4
DROPOUT = 0.0
LEARNING_RATE = 0.001
BATCH_SIZE = 512
EPOCHS = 1000
MAX_STEPS = 300

# Load dataset
dataset = PushTdataset(DATASET_PATH)

class ResBlock(nn.Module):
    def __init__(self, hidden_dim, dropout):
        super(ResBlock, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.layers(x) + x)

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, hidden_depth, dropout):
        super(MLP, self).__init__()
        layers = [nn.Linear(input_dim, hidden_dim), nn.ReLU()]
        for _ in range(hidden_depth - 1):
            layers.append(ResBlock(hidden_dim, dropout))
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

def train_mlp(model, dataset, device, writer, lr, batch_size, epochs):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            act = batch['action'].flatten(start_dim=1).to(device)
            obs = batch['obs'].flatten(start_dim=1).to(device)
            loss = criterion(model(obs), act)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)
        writer.add_scalar('Training Loss', avg_loss, epoch)
        if (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
    print("Training completed.")

def generate_path(model, device, stats, max_steps):
    env = PushTEnv()
    obs, _ = env.reset()
    rewards = []
    imgs = [env.render(mode='rgb_array')]
    model.eval()
    with torch.no_grad(), tqdm(total=max_steps, desc="Eval PushTStateEnv") as pbar:
        for step in range(max_steps):
            nobs = normalize_data(obs, stats=stats['obs'])[None, ...]
            nobs = torch.from_numpy(nobs).to(device, dtype=torch.float32)
            action_pred = model(nobs)
            nact = action_pred.cpu().numpy()[0]
            action_pred = unnormalize_data(nact, stats=stats['action'])
            action = obs[:2] + action_pred
            obs, reward, done, _, info = env.step(action)
            rewards.append(reward)
            pbar.update(1)
            pbar.set_postfix(reward=reward)
            imgs.append(env.render(mode='rgb_array'))
            if reward == 1 or done:
                break
    print('Score:', max(rewards))
    vwrite('vis_MLP.mp4', imgs)
    return max(rewards)

def success_rate(model, device, stats, max_steps):
    results = [generate_path(model, device, stats, max_steps) for _ in range(50)]
    return sum(r >= 0.8 for r in results) / 50, sum(r >= 0.9 for r in results) / 50

# Main execution
if __name__ == "__main__":
    model = MLP(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM, HIDDEN_DEPTH, DROPOUT).to(DEVICE)
    
    #writer = SummaryWriter("runs/MLP_training")
    #train_mlp(model, dataset, DEVICE, writer, LEARNING_RATE, BATCH_SIZE, epochs=EPOCHS)
    #writer.close()

    #torch.save(model.state_dict(), 'MLP_pushT.pth')
    #print(f"Model saved to MLP_pushT.pth")

    model.load_state_dict(torch.load('MLP_pushT.pth', map_location=DEVICE))
    model.eval()
    print(f"Model loaded from MLP_pushT.pth")

    #generate_path(model, DEVICE, dataset.stats, MAX_STEPS)
    result_80, result_90 = success_rate(model, DEVICE, dataset.stats, MAX_STEPS)
    print(f"Success rate (>=0.8): {result_80}, Success rate (>=0.9): {result_90}")