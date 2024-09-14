from typing import Tuple, Sequence, Dict, Union, Optional
import numpy as np
import math
import torch
import torch.nn as nn 
import collections
import zarr
from tqdm import tqdm

import gym
from gym import spaces
import pygame
import pymunk
from pymunk.space_debug_draw_options import SpaceDebugColor
from pymunk.vec2d import Vec2d
import shapely.geometry as sg
import cv2
import skimage.transform as st
from skvideo.io import vwrite
from IPython.display import Video
import gdown
import os
from IPython.display import Video
import pymunk.pygame_util
from torch.utils.data import DataLoader, TensorDataset
from methods import energy_discrepancy, langvin_sample
from torch import optim
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
writer = SummaryWriter(log_dir='runs/overfit')

def get_data_stats(data):
  data = data.reshape(-1, data.shape[-1])
  stats = {
    'min': np.min(data, axis=0),
    'max': np.max(data, axis=0),
  }
  return stats

# normalize
def normalize_data(data, stats):
  ndata = (data - stats['min'])/(stats['max'] - stats['min'])
  ndata = ndata * 2 - 1
  return ndata

def unnormalize_data(ndata, stats):
  ndata = (ndata + 1)/2
  data = ndata * (stats['max'] - stats['min']) + stats['min']
  return data

dataset_path = "dataset/pusht_cchi_v7_replay.zarr.zip"

class PushTdataset(torch.utils.data.Dataset):
    def __init__(self, dataset_path):
        dataset_root = zarr.open(dataset_path, 'r')
        train_data = {
            'action':dataset_root['data']['action'][:],
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
       '''if idx in self.episode_ends:
        nsample['next_obs'] = np.concatenate([nsample['action'], np.array([256, 256, np.pi / 4], dtype=np.float32)])
       else:
        nsample['next_obs'] = self.normalized_train_data['obs'][idx+1]'''
       if idx in self.episode_ends:
        nsample['next_obs'] = np.array([256, 256, np.pi / 4], dtype=np.float32)
       else:
        nsample['next_obs'] = self.normalized_train_data['obs'][idx+1]
       return nsample

class EBM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(EBM, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.act1 = nn.SiLU()
        self.fc2 = nn.Linear(hidden_size,hidden_size)
        self.act2 = nn.SiLU()
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.act3 = nn.SiLU()
        self.fc4 = nn.Linear(hidden_size, hidden_size)
        self.act4 = nn.SiLU()
        self.fc5 = nn.Linear(hidden_size, output_size)
        
    def forward(self, act, obs):
        x = torch.cat((act.float(), obs.float()), dim=-1)
        x = self.act1(self.fc1(x))
        x = self.act2(self.fc2(x))
        x = self.act3(self.fc3(x))
        x = self.act4(self.fc4(x))
        x = self.fc5(x)
        return x
    
dataset = PushTdataset(dataset_path)
stats = dataset.stats
dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=1,
    num_workers=1,
    shuffle=False,
    pin_memory=True,
    persistent_workers=True
)

batch = next(iter(dataloader))
print(batch['obs'])
print(batch['action'])
print(batch['next_obs'])

epochs = 1000
#data_one = dataset[:1]

def train_ebm(model, len_of_act, t_noise=0.2, num_epochs=500, batch_size=512):
    #dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    model = model.to(device)
    optimizer = optim.Adam(model.parameters())
    input = batch['obs'].to(device)
    target = torch.cat((batch['action'], batch['next_obs']), dim=-1).to(device)[:, :len_of_act]
    for epoch in range(num_epochs):
        #for batch in dataloader:
        #target = torch.cat((batch['action'], batch['next_obs'][:, :1]), dim=-1).to(device)
        #target = batch['action'].to(device)
        loss = energy_discrepancy(model, target, input, margin=None, t_noise=t_noise, m_particles=256, w_stable=1.0)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        

def draw_energy(model, nacts, nobs, record, path):
    model = model.cpu()
    #model.eval()
    x, y = torch.linspace(-1., 1., 100), torch.linspace(-1., 1., 100)
    X_mesh, Y_mesh = torch.meshgrid(x, y)
    X_mesh = X_mesh[...,None]
    Y_mesh = Y_mesh[...,None]
    p = torch.cat((X_mesh, Y_mesh), dim=-1).float().unsqueeze(2)
    temp = nobs[None, None, ...].repeat([100, 100, 1, 1])
    Z_mesh = model(p, temp.float()).detach().numpy().squeeze()
    i, j = np.unravel_index(np.argmin(Z_mesh), Z_mesh.shape)
    plt.figure()
    plt.contourf(X_mesh.numpy().squeeze(), Y_mesh.numpy().squeeze(), Z_mesh.squeeze(), level=30)
    #plt.imshow(np.fliplr(Z_mesh.squeeze()), extent=[-1, 1, -1, 1], cmap='hot', interpolation='nearest')
    plt.colorbar()
    plt.scatter(nacts[0], nacts[1], color='red', marker='x')
    plt.scatter(x[i], y[j] , color='blue', marker='o')
    plt.scatter(record[:, 0], record[:, 1], color='green', marker='.')
    plt.savefig(path)

nobs = batch['obs'].to(device)
with open('overfit.txt', 'w') as f:
    for t_noise in [0.001, 0.01, 0.03, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]: 
        for i in range(7): 
            target = torch.cat((batch['action'], batch['next_obs']), dim=-1).to(device)[:, :i+1]
            samples = []
            errors = []
            for epoch in range(50):
                ebm = EBM(6+i, 1024, 1)
                train_ebm(ebm, i+1, t_noise=t_noise)
                #draw_energy(ebm, batch['action'][0], batch['obs'], 'ebm1.png')
                #ebm = ebm.to(device)
                #act = torch.cat((batch['action'], batch['next_obs'][:, :1]), dim=-1)
                act = langvin_sample(ebm, target+0.05*torch.rand_like(target), nobs, e_l_step_size=0.02, n_iters=100, grad_decay=0.5, decay_step=10, noise_scale=0.0)
                samples.append(act.detach().cpu().numpy())
                print(f'predict:{act.detach().cpu().numpy()}')
                errors.append((target - act).abs().mean().detach().cpu().numpy())
                writer.add_scalar(f'dim:{i+1} t_noise{t_noise}', (target - act).abs().mean().detach().cpu().numpy(), epoch)
            samples = np.array(samples)
            errors = np.array(errors)
            writer.add_scalar(f'dim:{i+1}/error mean', np.mean(errors), t_noise*100)
            writer.add_scalar(f'dim:{i+1}/var', np.var(samples), t_noise*100)
            f.write(f'dim:{i+1} t_noise:{t_noise} samples:{samples} truth:{target} var:{np.var(samples)} error mean:{np.mean(errors)}')
            f.write('\n')
            
#draw_energy(ebm, batch['action'][0], batch['obs'], record.squeeze(), 'ebm2.png')