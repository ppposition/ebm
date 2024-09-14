from environments.pushT import PushTEnv
import torch
from tqdm import tqdm
from methods import langvin_sample, DFO_infer
from dataset.PushTdataset import normalize_data, unnormalize_data
from skvideo.io import vwrite
from IPython.display import Video
import numpy as np
import argparse
import yaml 
from model.net import MLP, MLP_cond, TransformerModel
import collections
import pickle
import os

def generate_path(model, device, stats, path, action_dim, pred_horizon, action_horizon, obs_horizon, obs_dim, sample_method, sample_dic, max_steps=300):
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
            if sample_method=="DFO": 
                act = DFO_infer(model, obs_cond, pred_horizon=pred_horizon,
                action_dim=action_dim, **sample_dic)
            elif sample_method=="Langevin":
                act = langvin_sample(model, act, obs_cond, **sample_dic)
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
    vwrite(os.path.join(path, f'vis.mp4'), imgs)
    #from Ipython.display import Video
    '''if max(rewards)<0.8:
        order = len(os.listdir(path))
        vwrite(os.path.join(path, f'vis{order}.mp4'), imgs)'''
    Video('vis.mp4', embed=True, width=256, height=256)
    return max(rewards)

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str)
    parser.add_argument('--sample', type=str)
    parser.add_argument('--stat', type=str)
    parser.add_argument('--net', type=str)
    parser.add_argument('--parameter', type=str)
    parser.add_argument('--path', type=str)
    
    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    with open(args.stat, 'rb') as f:
        stat = pickle.load(f)
        
    with open('configs/'+args.task+'.yaml', 'r') as file:
        data_config = yaml.safe_load(file)['datashape']
        
    with open('configs/'+args.sample+'.yaml', 'r') as file:
        sample_config = yaml.safe_load(file)
        
    with open('configs/models.yaml', 'r') as file:
        model_config = yaml.safe_load(file)[args.net]
        if args.net=='MLP':
            model = MLP(input_dim=data_config['obs_horizon']*data_config['obs_dim']+2*data_config['pred_horizon']*data_config['action_dim'], **model_config)
        elif args.net=='MLP_cond':
            model = MLP_cond(input_dim=data_config['pred_horizon']*data_config['action_dim'],cond_dim=data_config['obs_horizon']*data_config['obs_dim'], **model_config)
    model.load_state_dict(torch.load(args.parameter))
    if not os.path.exists(args.path):
        os.mkdir(args.path)
    result_80 = 0
    result_90 = 0
    for i in range(50):
        reward = generate_path(model, device, stat, args.path, **data_config, sample_method=args.sample, sample_dic=sample_config)
        if reward>=0.8:
            result_80 += 1
            if reward>=0.9:
                result_90 += 1
    with open(os.path.join(args.path,'result.txt'), 'w') as f:
        f.write("noise_scale:{}\n".format(sample_config['noise_scale']))
        f.write("rate_80:"+str(result_80/50)+'\n')
        f.write("rate_90:"+str(result_90/50))
    