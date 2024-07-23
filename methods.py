import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
from torch.utils.data import random_split

def langvin_sample_step(net, act, obs, e_l_step_size, noise_scale): 
    net.eval()
    act = act.clone().detach().requires_grad_(True)
    en = net(act, obs)
    grad = torch.autograd.grad(en.sum(), act, allow_unused=True)[0]
    act.data = act.data - 0.5 * e_l_step_size**2 * grad + e_l_step_size * noise_scale * torch.randn_like(act.data)
    act.data = torch.clip(act.data, -1, 1)
    return act, en

def langvin_sample(net, act, obs, e_l_step_size, n_iters, grad_decay, decay_step, noise_scale):
    for iter in range(1, n_iters+1):
        if iter%decay_step==0:
            e_l_step_size *= grad_decay
        act, _ = langvin_sample_step(net, act, obs, e_l_step_size, noise_scale)
    return act

def energy_discrepancy(energy_net, act, obs, m_particles=16, t_noise=0.5, w_stable=1.0):
    device = act.device
    browniani = torch.randn_like(act).to(device) * t_noise
    brownianij = torch.randn(act.size(0), m_particles, *act.shape[1:]).to(device) * t_noise
    pert_data = act.unsqueeze(1) + browniani.unsqueeze(1) + brownianij
    pert_data = torch.clip(pert_data, -1, 1)
    obs_mul = obs.unsqueeze(1) + (torch.zeros(obs.size(0), m_particles, *obs.shape[1:])).to(device)
    pos_energy = energy_net(act=act, obs=obs)
    neg_energy = energy_net(act=pert_data.view(-1, *act.shape[1:]), obs=obs_mul.view(-1, *obs.shape[1:]).flatten(start_dim=1)).view(act.size(0), -1)
    val = pos_energy - neg_energy
    if w_stable != 0:
        val = torch.cat([val, np.log(w_stable) * torch.ones_like(val[:, :1])], dim=-1)
    loss = val.logsumexp(dim=-1).mean()
    return loss, pos_energy.max(), neg_energy.max()

def energy_discrepancy_train(model, dataset, device, path, m_particles=16, t_noise=0.5, w_stable=1.0, decay=1, epochs=100, lr=0.01, gamma=0.99, batch_size=256, e_l_step_size=0.2, n_iters=100, grad_decay=0.5, decay_step=10, noise_scale=1.0):
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
        batch_size=1,
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
            loss, max_pos_energy, max_neg_energy = energy_discrepancy(model, act, obs, m_particles=m_particles, t_noise=t_noise, w_stable=w_stable)
            optim.zero_grad()
            loss.backward()
            optim.step()
            lr_schedule.step()
            pbar.set_description('Epoch:{:4d},loss:{:.8f},pos_energy:{:.4f}'.format(epoch, loss.item(), max_pos_energy.item()))
            ED.append(loss.cpu().detach().numpy())
        test_loss = 0.0
        model.eval()
        for batch in test_dataloader:      
            a = langvin_sample(model, torch.randn_like(act[:1]).to(device), obs[:1], e_l_step_size, n_iters, grad_decay, decay_step, noise_scale)
            test_loss += torch.norm(a - act[:1]).cpu().detach().numpy()
        test_loss = test_loss/len(test_dataloader)
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
        f.write('ED:{:.4f}'.format(ED[-1]))