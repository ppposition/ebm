import argparse 
import yaml
from model.net import MLP, MLP_cond, TransformerModel, SemiUnet
from dataset.PushTdataset import PushTStateDataset
import torch
from tqdm import tqdm
from methods import energy_discrepancy_train, langvin_sample
from functools import partial
import os
import numpy as np
import pickle
from dataset.PushTdataset import normalize_data, unnormalize_data
from environments.pushT import PushTEnv
import collections
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, random_split

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dataset = PushTStateDataset("dataset/pusht_cchi_v7_replay.zarr.zip", -1, pred_horizon=1, obs_horizon=1, action_horizon=1)


def generate_path(model, device, stats, action_dim, pred_horizon, action_horizon, obs_horizon, max_steps=300):
    env = PushTEnv()
    #env.seed(10000)
    obs, inf = env.reset()
    
    obs_deque = collections.deque(
        [obs] * obs_horizon, maxlen=obs_horizon
    )
    imgs = [env.render(mode='rgb_array')]
    rewards = list()
    done = False
    step_idx=0
    act = torch.randn((1, pred_horizon, action_dim), device=device)
    model = model.to(device)
    model.eval()
    with tqdm(total=max_steps, desc="Eval PushTStateEnv") as pbar:
        while not done:
            obs_seq = np.stack(obs_deque)
            nobs = normalize_data(obs_seq, stats=stats['obs'])
            nobs = torch.from_numpy(nobs).to(device, dtype=torch.float32)
            obs_cond = nobs.unsqueeze(0)
            act = langvin_sample(model, act, obs_cond)
            nact = act.detach().to('cpu').numpy()
            nact = nact[0]
            action_pred = unnormalize_data(nact, stats=stats['action'])
            start = obs_horizon-1
            end = start + action_horizon
            action = action_pred[start:end, :]
            for i in range(len(action)):
                obs, reward, done, _, info = env.step(action[i])
                obs_deque.append(obs)
                rewards.append(reward)
                imgs.append(env.render(mode='rgb_array'))

                step_idx += 1
                pbar.update(1)
                pbar.set_postfix(reward=reward)
                if reward==1:
                    print(obs)
                if step_idx>max_steps:
                    done = True
                if done:
                    break
                
    print('Score:', max(rewards))
    #from Ipython.display import Video
    return max(rewards)

def success_rate(model, device, stats, action_dim, pred_horizon, action_horizon, obs_horizon, max_steps=300):
    result_80, result_90 = 0, 0
    for i in range(50):
        reward = generate_path(model, device, stats, action_dim, pred_horizon, action_horizon, obs_horizon, max_steps=max_steps)
        if reward>=0.8:
            result_80 += 1
            if reward>=0.9:
                result_90 += 1
    return result_80/50, result_90/50

def test_error(model, test_dataset):
    test_dataloader = DataLoader(test_dataset, batch_size=256, shuffle=False)
    error = 0
    for batch in test_dataloader:
        target = batch['action'].to(device)
        nobs = batch['obs'].to(device)
        act = langvin_sample(model, target+0.05*torch.rand_like(target), nobs, e_l_step_size=0.02, n_iters=100, grad_decay=0.5, decay_step=10, noise_scale=0.0)
        error += torch.mean(torch.abs(act-target)).item()
    return error/len(test_dataloader)

writer = SummaryWriter('runs/test_error')
dataset_size = len(dataset)
train_size = int(0.8 * dataset_size)  # 80% 用作训练集
test_size = dataset_size - train_size  # 20% 用作测试集
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

with open('noise_w.txt', 'w') as f:
    for i, t_noise in enumerate([0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6]):
        for w_stable in [0.2, 0.5, 1.0, 2.0]:
            model = MLP(input_dim=7, hidden_dim=512, output_dim=1, hidden_depth=4, dropout=0).to(device)
            #testError = test_error(model, test_dataset)
            energy_discrepancy_train(model, train_dataset, device, writer=writer, lr=0.001, gamma=0.99, batch_size=512, epochs=1000, t_noise=t_noise, w_stable=w_stable)
            testError = test_error(model, test_dataset)
            print(f't_noise:{t_noise} w:{w_stable} test_error:{testError}' )
            writer.add_scalar(f'w:{w_stable}/test_error', testError, i)
            result_80, result_90 = success_rate(model, device, dataset.stats, 2, dataset.pred_horizon, dataset.action_horizon, dataset.obs_horizon, max_steps=300)
            f.write(f't_noise:{t_noise} w:{w_stable} rate_80:{result_80} rate_90:{result_90}\n')
            print(f't_noise:{t_noise} w:{w_stable} rate_80:{result_80} rate_90:{result_90}\n')
            f.write(f't_noise:{t_noise} w:{w_stable} test_error:{test_error(model, test_dataset)}' )
            writer.add_scalar(f'w:{w_stable}/result_80', result_80, i)
            writer.add_scalar(f'w:{w_stable}/result_90', result_90, i)
            
writer.close()
        