import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
from torch.utils.data import random_split
import torch.nn.functional as F

def reflect(tensor, down, up):
    t = tensor.clone()
    while ((down<t) & (t>up)).any():   
        t = torch.where(t<down, 2*down-t, t)
        t = torch.where(t>up, 2*up-t, t)
    return t

def langvin_sample_step(net, act, obs, margin, e_l_step_size, noise_scale): 
    net.eval()
    act = act.clone().detach().requires_grad_(True)
    en = net(act, obs)
    grad = torch.autograd.grad(en.sum(), act, allow_unused=True)[0]
    act.data = act.data - 0.5 * e_l_step_size**2 * grad + e_l_step_size * noise_scale * torch.randn_like(act.data)
    if margin=='clip':
        act.data = torch.clip(act.data, -1, 1)
    elif margin=='reflect':
        act.data = reflect(act.data, -1, 1)
    return act, en

def langvin_sample(net, act, obs, margin, e_l_step_size, n_iters, grad_decay, decay_step, noise_scale):
    for iter in range(1, n_iters+1):
        if iter%decay_step==0:
            e_l_step_size *= grad_decay
        act, _ = langvin_sample_step(net, act, obs, margin, e_l_step_size, noise_scale)
    return act

def energy_discrepancy(energy_net, act, obs, margin, m_particles=16, t_noise=0.5, w_stable=1.0):
    device = act.device
    browniani = torch.randn_like(act).to(device) * t_noise
    brownianij = torch.randn(act.size(0), m_particles, *act.shape[1:]).to(device) * t_noise
    pert_data_origin = act.unsqueeze(1) + browniani.unsqueeze(1) + brownianij
    if margin=='clip':
        pert_data = torch.clip(pert_data_origin, -1, 1)
    elif margin=='reflect':
        pert_data = reflect(pert_data_origin, -1, 1)
    obs_mul = obs.unsqueeze(1) + (torch.zeros(obs.size(0), m_particles, *obs.shape[1:])).to(device)
    pos_energy = energy_net(act=act, obs=obs)
    neg_energy = energy_net(act=pert_data, obs=obs_mul)
    val = pos_energy - neg_energy.squeeze()
    if w_stable != 0:
        val = torch.cat([val, np.log(w_stable) * torch.ones_like(val[:, :1])], dim=-1)
    loss = val.logsumexp(dim=-1).mean()
    return loss, pos_energy.max(), neg_energy.max()

def energy_discrepancy_train(model, dataset, device, path, margin='clip', m_particles=16, t_noise=0.5, w_stable=1.0, decay=1, epochs=100, lr=0.01, gamma=0.99, batch_size=256, e_l_step_size=0.2, n_iters=100, grad_decay=0.5, decay_step=10, noise_scale=1.0):
    model = model.to(device)
    test_size = len(dataset)//100
    train_size = len(dataset) - test_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=1,
        shuffle=True,
        pin_memory=True,
        persistent_workers=True
    )
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        num_workers=1,
        shuffle=False,
    )
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    lr_schedule = torch.optim.lr_scheduler.ExponentialLR(optim, gamma=gamma)
    mse_loss = []
    ED = []
    for epoch in range(epochs):
        model.train()
        pbar = tqdm(train_dataloader)
        for batch in pbar:
            act, obs = batch['action'].to(device), batch['obs'].to(device)
            model.train()
            loss, max_pos_energy, max_neg_energy = energy_discrepancy(model, act, obs, margin, m_particles=m_particles, t_noise=t_noise, w_stable=w_stable)
            optim.zero_grad()
            loss.backward()
            optim.step()
            lr_schedule.step()
            pbar.set_description('Epoch:{:4d},loss:{:.8f},pos_energy:{:.4f}'.format(epoch, loss.item(), max_pos_energy.item()))
            ED.append(loss.cpu().detach().numpy())
        test_loss = 0.0
        model.eval()
        num_test_samples = 0
        for batch in test_dataloader:
            act, obs = batch['action'].to(device), batch['obs'].to(device)    
            a = langvin_sample(model, torch.randn_like(act).to(device), obs, margin, e_l_step_size, n_iters, grad_decay, decay_step, noise_scale)
            test_loss += torch.norm(a - act, dim=list(range(1, len(a.shape)))).cpu().detach().numpy().sum()
            num_test_samples += act.shape[0]*act.shape[1]
        test_loss = test_loss/num_test_samples
        print("test error:{:.4f}".format(test_loss))
        mse_loss.append(test_loss)
        t_noise *= decay
    plt.plot(mse_loss)
    plt.savefig(os.path.join(path, 'test_loss.png'))
    plt.cla()
    plt.plot(ED)
    plt.savefig(os.path.join(path, 'ED.png'))
    with open(os.path.join(path, 'result.txt'), 'w') as f:
        f.write('test_loss:{:.8f}\n'.format(mse_loss[-1]))
        f.write('ED:{:.4f}\n'.format(ED[-1]))
        
def DFO(model, act, obs, num_samples):
    device = act.device
    negatives = torch.rand(act.size(0), num_samples, *act.shape[1:]).to(device) * 2 - 1
    acts = torch.cat([act.unsqueeze(dim=1), negatives], dim=1)
    permutation = torch.rand(acts.size(0), acts.size(1)).argsort(dim=1)
    acts = acts[torch.arange(acts.size(0)).unsqueeze(-1), permutation]
    ground_truth = (permutation==0).nonzero()[:, 1].to(device)
    obs = obs.unsqueeze(1)
    obs = obs.repeat(1, acts.shape[1], 1, 1)
    energy = model(acts, obs).squeeze()
    logits = -1.0 * energy
    loss = F.cross_entropy(logits, ground_truth)
    return loss

def DFO_train(model, dataset, device, path, sample_method, sample_dic,num_samples=128, epochs=100, lr=0.001, gamma=0.99, batch_size=256):
    model = model.to(device)
    test_size = len(dataset)//100
    train_size = len(dataset) - test_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=1,
        shuffle=True,
        pin_memory=True,
        persistent_workers=True
    )
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=256,
        num_workers=1,
        shuffle=False,
    )
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    lr_schedule = torch.optim.lr_scheduler.ExponentialLR(optim, gamma=gamma)
    mse_loss = []
    ED = []
    for epoch in range(epochs):
        model.train()
        pbar = tqdm(train_dataloader)
        for batch in pbar:
            act, obs = batch['action'].to(device), batch['obs'].to(device)
            model.train()
            loss = DFO(model, act, obs, num_samples)
            optim.zero_grad()
            loss.backward()
            optim.step()
            lr_schedule.step()
            pbar.set_description('Epoch:{:4d},loss:{:.8f}'.format(epoch, loss.item()))
            ED.append(loss.cpu().detach().numpy())
        test_loss = 0.0
        model.eval()
        num_test_samples=0
        for batch in test_dataloader:
            act, obs = batch['action'].to(device), batch['obs'].to(device)
            pred_horizon, action_dim = act.shape[-2], act.shape[-1]
            if sample_method=="DFO": 
                a_pre = DFO_infer(model, obs, pred_horizon=pred_horizon,
                action_dim=action_dim, **sample_dic)
            elif sample_method=="Langevin":
                a_pre = langvin_sample(model, act, obs, **sample_dic)
            test_loss += torch.norm(a_pre - act, dim=list(range(1, len(a_pre.shape)))).cpu().detach().numpy().sum()
            num_test_samples += act.shape[0]*act.shape[1]
        test_loss = test_loss/num_test_samples
        print("test error:{:.4f}".format(test_loss))
        mse_loss.append(test_loss)
    plt.plot(mse_loss)
    plt.savefig(os.path.join(path, 'test_loss.png'))
    plt.cla()
    plt.plot(ED)
    plt.savefig(os.path.join(path, 'ED.png'))
    with open(os.path.join(path, 'result.txt'), 'w') as f:
        f.write('test_loss:{:.8f}\n'.format(mse_loss[-1]))
        f.write('ED:{:.4f}\n'.format(ED[-1]))
        
def DFO_infer(ebm, obs, noise_scale, inference_samples, iters, pred_horizon, action_dim, noise_shrink):
    samples = torch.rand(obs.size(0), inference_samples, pred_horizon, action_dim).to(obs.device) * 2 - 1
    obs = obs.unsqueeze(1)
    obs = obs.repeat(1, inference_samples, 1, 1)
    for _ in range(iters):
        energies = ebm(samples, obs).squeeze()
        probs = F.softmax(-1.0*energies, dim=-1)
        idxs = torch.multinomial(probs, inference_samples, replacement=True)
        samples = samples[torch.arange(samples.size(0)).unsqueeze(-1), idxs]
        samples = samples.clamp(-1, 1)
        noise_scale *= noise_shrink
    
    energies = ebm(samples, obs).squeeze()
    probs = F.softmax(-1.0*energies, dim=-1)
    best_idxs = probs.argmax(dim=-1)
    return samples[torch.arange(samples.size(0)), best_idxs, :]