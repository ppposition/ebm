import argparse 
import yaml
from model.net import MLP, MLP_cond, TransformerModel, SemiUnet, MLP_Conv
from dataset.PushTdataset import PushTStateDataset
import torch
from tqdm import tqdm
from methods import energy_discrepancy_train, langvin_sample, DFO_train, DFO_infer
from functools import partial
import os
import numpy as np
import pickle
from test import generate_path


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str)
    parser.add_argument('--net', type=str)
    parser.add_argument('--method', type=str)
    parser.add_argument('--sample', type=str)
    parser.add_argument('--sample_steps', type=int)
    args = parser.parse_args()
    dir_path = os.path.join('result',str(len(os.listdir('result'))+1))
    os.mkdir(dir_path)
    with open('configs/'+args.task+'.yaml', 'r') as file:
        data_config = yaml.safe_load(file)
        datashape_config = data_config['datashape']
        assert datashape_config['pred_horizon'] >= datashape_config['action_horizon']
        dataset = PushTStateDataset(data_config['path'], data_config['path_size'], pred_horizon=datashape_config['pred_horizon'], obs_horizon=datashape_config['obs_horizon'], action_horizon=datashape_config['action_horizon'])
    with open('configs/models.yaml', 'r') as file:
        model_config = yaml.safe_load(file)[args.net]
        if args.net=='MLP':
            model = MLP(input_dim=datashape_config['obs_horizon']*datashape_config['obs_dim']+2*datashape_config['pred_horizon']*datashape_config['action_dim'], **model_config)
        elif args.net=='MLP_cond':
            model = MLP_cond(input_dim=datashape_config['pred_horizon']*datashape_config['action_dim'],cond_dim=datashape_config['obs_horizon']*datashape_config['obs_dim'], **model_config)
        elif args.net=='Unet_1d':
            model = SemiUnet(input_dim=datashape_config['action_dim'], global_cond_dim=datashape_config['obs_horizon']*datashape_config['obs_dim'], **model_config, len_seq=datashape_config['pred_horizon'])
        elif args.net=='MLP_Conv':
            model = MLP_Conv(input_dim=datashape_config['obs_horizon']*datashape_config['obs_dim']+datashape_config['action_dim'], action_dim=datashape_config['action_dim'], **model_config)
        else:
            raise NameError('No such model')
    with open('configs/train.yaml', 'r') as file:
        train_config = yaml.safe_load(file)
    
    with open('configs/train_method.yaml', 'r') as file:
        EBM_config = yaml.safe_load(file)[args.method]
    
    with open('configs/'+args.sample+'.yaml', 'r') as file:
        sample_config = yaml.safe_load(file)
        
    with open(dir_path+'/parameter.txt', 'w') as file:
        yaml.dump({'task':args.task}, file)
        yaml.dump(data_config,file)
        file.write('\n')
        yaml.dump({'model':args.net}, file)
        yaml.dump(model_config, file)
        file.write('\n') 
        yaml.dump({'method':args.method}, file)
        yaml.dump(EBM_config, file)
        file.write('\n') 
        yaml.dump(train_config, file)
        file.write('\n')
        yaml.dump({'sample method':args.sample}, file)
        yaml.dump(sample_config, file)
        
    stats = dataset.stats
    with open(os.path.join(dir_path, 'd1.pkl'), 'wb') as f:
        pickle.dump(stats, f)
        
    if args.method=='ED':
        energy_discrepancy_train(model, dataset, device, dir_path, **EBM_config, **train_config)
    elif args.method=='DFO':
        DFO_train(model, dataset, device, dir_path, sample_method=args.sample, sample_dic=sample_config, **EBM_config, **train_config)
    torch.save(model.state_dict(), os.path.join(dir_path, 'ebm.pth')) 
    rewards = generate_path(model, device, stats, dir_path, **datashape_config, sample_method=args.sample, sample_dic=sample_config, max_steps=args.sample_steps)
    with open(os.path.join(dir_path, 'result.txt'), 'a') as f:
        f.write('reward:{}\n'.format(rewards))
    
    