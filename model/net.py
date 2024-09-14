import torch
import torch.nn as nn
from functools import partial
import torch.nn.functional as F
import torchvision
from typing import Callable

'''class MLP(nn.Module):
    def __init__(self, input_dim:int, hidden_dim:int, hidden_depth:int, output_dim:int, dropout:float) -> None:
        super().__init__()
        dropout_layer = partial(nn.Dropout, p=dropout)
        layers1 = [nn.Linear(input_dim, hidden_dim), nn.Mish(), dropout_layer()]
        layers2 = [nn.Linear(input_dim, hidden_dim), nn.Mish(), dropout_layer()]
        for _ in range(hidden_depth-1):
            layers1 += [nn.Linear(hidden_dim, hidden_dim), nn.Mish(), dropout_layer()]
            layers1 += [nn.Linear(hidden_dim, hidden_dim), nn.Mish(), dropout_layer()]
        layers1 += [nn.Linear(hidden_dim, input_dim)]
        layers2 += [nn.Linear(hidden_dim, input_dim)]
        self.layers1 = nn.Sequential(*layers1)
        self.layers2 = nn.Sequential(*layers2)
        self.linear = nn.Linear(input_dim, output_dim)
    def forward(self, act, obs):
        x = torch.cat((act.flatten(start_dim=-2), obs.flatten(start_dim=-2)), dim=-1)
        output1 = self.layers1(x) + x
        output = output1 + self.layers2(output1)
        return self.linear(output)'''

class MLP(nn.Module):
    def __init__(self, input_dim:int, hidden_dim:int, hidden_depth:int, output_dim:int, dropout:float) -> None:
        super().__init__()
        dropout_layer = partial(nn.Dropout, p=dropout)
        layers1 = [nn.Linear(input_dim, hidden_dim), nn.SiLU(), dropout_layer()]
        for _ in range(hidden_depth-1):
            layers1 += [nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), dropout_layer()]
        layers1.append(nn.Linear(hidden_dim, output_dim))
        self.layers1 = nn.Sequential(*layers1)
    def forward(self, act, obs):
        x = torch.cat((act.flatten(start_dim=-2), act.flatten(start_dim=-2), obs.flatten(start_dim=-2)), dim=-1)
        return self.layers1(x)
    
class Conv1dBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kerner_size, n_groups=8) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=kerner_size, stride=1, padding=kerner_size//2),
            nn.GroupNorm(n_groups, out_channels),
            nn.SiLU()
        )
    def forward(self, x):
        return self.block(x)
    
class MLP_Conv(nn.Module):
    def __init__(self, input_dim:int, hidden_channel, hidden_dim:int, hidden_depth:int, output_dim:int, action_dim:int, dropout:float) -> None:
        super().__init__()
        dropout_layer = partial(nn.Dropout, p=dropout)
        self.conv1 = Conv1dBlock(action_dim, hidden_channel, 3)
        self.residual_conv = nn.Conv1d(hidden_channel, hidden_channel, 1) if action_dim != hidden_channel else nn.Identity()
        self.downsample = nn.Conv1d(hidden_channel, action_dim, 3, 2, 1)
        MLP_layers = [nn.Linear(input_dim, hidden_dim), nn.SiLU(), dropout_layer()]
        for _ in range(hidden_depth-1):
            MLP_layers += [nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), dropout_layer()]
        MLP_layers.append(nn.Linear(hidden_dim, output_dim))
        self.MLP = nn.Sequential(*MLP_layers)
        
    def forward(self, act, obs):
        act = act.moveaxis(-1, -2)
        
        act_out = self.downsample(self.residual_conv(self.conv1(act))).flatten(start_dim=1)
        x = torch.cat((act_out.flatten(start_dim=1), obs.flatten(start_dim=1)), dim=1)
        return self.MLP(x)

class MLP_cond(nn.Module):
    def __init__(self, input_dim:int, hidden_dim:int, hidden_depth:int, output_dim:int, dropout:float, cond_dim:int, ) -> None:
        super().__init__()
        self.conv1 = Conv1dBlock()
        dropout_layer = partial(nn.Dropout, p=dropout)
        layers1 = [nn.Linear(input_dim, hidden_dim), nn.Mish(), dropout_layer()]
        layers2 = [nn.Linear(input_dim, hidden_dim), nn.Mish(), dropout_layer()]
        for _ in range(hidden_depth-1):
            layers1 += [nn.Linear(hidden_dim, hidden_dim), nn.Mish(), dropout_layer()]
            layers1 += [nn.Linear(hidden_dim, hidden_dim), nn.Mish(), dropout_layer()]
        layers1 += [nn.Linear(hidden_dim, input_dim)]
        layers2 += [nn.Linear(hidden_dim, input_dim)]
        self.layers1 = nn.Sequential(*layers1)
        self.layers2 = nn.Sequential(*layers2)
        self.linear = nn.Linear(input_dim, output_dim)
        self.cond_linear1 = nn.Sequential(
            nn.Mish(),
            nn.Linear(cond_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, input_dim*2)
        )
        self.cond_linear2 = nn.Sequential(
            nn.Mish(),
            nn.Linear(cond_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, input_dim*2)
        )
    def forward(self, act, obs):
        output1 = self.layers1(act) + act
        cond1 = self.cond_linear1(obs)
        output1 = cond1[:, :act.shape[1]]*output + cond1[:, act.shape[1]+1:]
        output = self.layers2(output1) + output1
        cond2 = self.cond_linear2(obs)
        output = cond2[:, :act.shape[1]]*output + cond2[:, act.shape[1]+1:]
        return self.linear(output)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.encoding = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(torch.log(torch.tensor(10000.0)) / d_model))
        self.encoding[:, 0::2] = torch.sin(position * div_term)
        self.encoding[:, 1::2] = torch.cos(position * div_term)
        self.encoding = self.encoding.unsqueeze(0)

    def forward(self, x):
        # Make sure input is of the correct size
        x = x * torch.sqrt(torch.tensor(self.encoding.size(2)))
        # Add positional encoding
        x = x + self.encoding[:, :x.size(1)].to(x.device)
        return x
    
class TransformerModel(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_encoder_layers, dim_feedforward, dropout=0.1):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        self.positional_encoding = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, activation=F.mish)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        self.output_layer = nn.Linear(d_model, input_dim)
        self.linear = nn.Linear(7, 1)
        
    def forward(self, act, obs):
        src = torch.cat((act, obs), dim=1)[..., None]     
        src = self.embedding(src)
        src = self.positional_encoding(src)
        src = src.permute(1, 0, 2)
        output = self.transformer_encoder(src)
        output = self.output_layer(output)
        output = output.permute(1, 0, 2)
        output = self.linear(output.squeeze(-1))
        return output

class Downsample1d(nn.Module):
    def __init__(self, dim) -> None:
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, 3, 2, 1)
    
    def forward(self, x):
        return self.conv(x)
    
class Conv1dBlock(nn.Module):
    def __init__(self, inp_channels, out_channels, kernel_size, n_groups=8) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(inp_channels, out_channels, kernel_size, padding=kernel_size//2),
            nn.GroupNorm(n_groups, out_channels),
            nn.Mish(),
        )
    
    def forward(self, x):
        return self.block(x)

class ConditionResidualBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, cond_dim, kernel_size=3, n_groups=8):
        super().__init__()
        self.blocks = nn.ModuleList(
            [Conv1dBlock(in_channels, out_channels, kernel_size, n_groups),
            Conv1dBlock(out_channels, out_channels, kernel_size, n_groups)]
        )
        cond_channels = out_channels * 2
        self.out_channels = out_channels
        self.cond_encoder = nn.Sequential(
            nn.Mish(),
            nn.Linear(cond_dim, cond_channels),
            nn.Unflatten(-1, (-1, 1))
        )
        self.residual_conv = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
    
    def forward(self, x, cond):
        out = self.blocks[0](x)
        cond = cond.flatten(start_dim=1)
        embed = self.cond_encoder(cond)
        embed = embed.reshape(embed.shape[0], 2, self.out_channels, 1)
        scale = embed[:, 0, ...]
        bias = embed[:, 1, ...]
        out = scale * out + bias

        out = self.blocks[1](out)
        out = out + self.residual_conv(x)
        return out

class SemiUnet(nn.Module):
    def __init__(self, input_dim, global_cond_dim, down_dims=[256, 512, 1024], kernel_size=5, n_groups=8, len_seq=16, hidden_layer=1000) -> None:
        super().__init__()
        all_dims = [input_dim] + list(down_dims)
        cond_dim = global_cond_dim
        in_out = list(zip(all_dims[:-1], all_dims[1:]))
        
        down_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (len(in_out) - 1)
            down_modules.append(nn.ModuleList([
                ConditionResidualBlock1D(
                    dim_in, dim_out, cond_dim=cond_dim,
                    kernel_size=kernel_size, n_groups=n_groups
                ),
                ConditionResidualBlock1D(
                    dim_out, dim_out, cond_dim=cond_dim,
                    kernel_size=kernel_size, n_groups=n_groups
                ),
                Downsample1d(dim_out) if not is_last else nn.Identity()
            ]))
        
        self.down_modules = down_modules
        self.linear1 = nn.Linear(down_dims[-1]*(len_seq//(2**(len(down_dims)-1
        ))), hidden_layer)
        self.activate_func = nn.Mish()
        self.linear2 = nn.Linear(hidden_layer, 1)
        print("number of parameters:{:e}".format(
            sum(p.numel() for p in self.parameters())
        ))
    
    def forward(self, act: torch.Tensor, obs=None):
        act = act.moveaxis(-1, -2)
        x = act
        for idx, (resnet, resnet2, downsample) in enumerate(self.down_modules):
            x = resnet(x, obs)
            x = resnet2(x, obs)
            x = downsample(x)
        x = self.linear2(self.activate_func(self.linear1(x.flatten(start_dim=1))))
        return x

def get_resnet(name:str, weights=None, **kwargs) -> nn.Module:
    func = getattr(torchvision.models, name)
    resnet = func(weights=weights, **kwargs)
    resnet.fc = torch.nn.Identity()
    return resnet

def replace_submodules(
        root_module: nn.Module,
        predicate: Callable[[nn.Module], bool],
        func: Callable[[nn.Module], nn.Module]) -> nn.Module:
    """
    Replace all submodules selected by the predicate with
    the output of func.

    predicate: Return true if the module is to be replaced.
    func: Return new module to use.
    """
    if predicate(root_module):
        return func(root_module)

    bn_list = [k.split('.') for k, m
        in root_module.named_modules(remove_duplicate=True)
        if predicate(m)]
    for *parent, k in bn_list:
        parent_module = root_module
        if len(parent) > 0:
            parent_module = root_module.get_submodule('.'.join(parent))
        if isinstance(parent_module, nn.Sequential):
            src_module = parent_module[int(k)]
        else:
            src_module = getattr(parent_module, k)
        tgt_module = func(src_module)
        if isinstance(parent_module, nn.Sequential):
            parent_module[int(k)] = tgt_module
        else:
            setattr(parent_module, k, tgt_module)
    # verify that all modules are replaced
    bn_list = [k.split('.') for k, m
        in root_module.named_modules(remove_duplicate=True)
        if predicate(m)]
    assert len(bn_list) == 0
    return root_module

def replace_bn_with_gn(
    root_module: nn.Module,
    features_per_group: int=16) -> nn.Module:
    """
    Relace all BatchNorm layers with GroupNorm.
    """
    replace_submodules(
        root_module=root_module,
        predicate=lambda x: isinstance(x, nn.BatchNorm2d),
        func=lambda x: nn.GroupNorm(
            num_groups=x.num_features//features_per_group,
            num_channels=x.num_features)
    )
    return root_module

class EBMConvMLP(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.cnn = get_resnet('resnet18', weights=None)
        self.linear = nn.Linear(512, 5)
        self.mlp = MLP(**config['mlp_config'])
        
    def forward(self, act, obs):
        
        feature = self.linear(self.cnn(obs.flatten(end_dim=1))).reshape(obs.shape[0], -1)
        return self.mlp(torch.cat([act.flatten(-2), feature], dim=-1)) 