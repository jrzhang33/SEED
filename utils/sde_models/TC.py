import torch
import torch.nn as nn
import numpy as np
from .attention import Seq_Transformer
from networks import Adver_network,common_network
class Discriminator(nn.Module):
    def __init__(self, input_dim=256, hidden_dim=256, num_domains=4):
        super(Discriminator, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        layers = [
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_domains),
        ]
        self.layers = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

class TC(nn.Module):
    def __init__(self, configs, device):
        super(TC, self).__init__()
        self.num_channels = configs.final_out_channels
        self.timestep = configs.TC.timesteps
        self.Wk = nn.ModuleList([nn.Linear(configs.TC.hidden_dim, self.num_channels) for i in range(self.timestep)])
        self.lsoftmax = nn.LogSoftmax()
        self.device = device
        
        self.projection_head1= nn.Sequential(
            nn.Linear(configs.TC.hidden_dim, configs.final_out_channels // 2),
            nn.BatchNorm1d(configs.final_out_channels // 2),
            nn.ReLU(inplace=True),
            nn.Linear(configs.final_out_channels // 2, configs.final_out_channels // 4),
        )
        
        self.projection_head2 = nn.Sequential(
            nn.Linear(configs.TC.hidden_dim, configs.final_out_channels // 2),
            nn.BatchNorm1d(configs.final_out_channels // 2),
            nn.ReLU(inplace=True),
            nn.Linear(configs.final_out_channels // 2, configs.final_out_channels // 4),
        )
        self.discriminator= Discriminator(input_dim=configs.final_out_channels // 4, hidden_dim=256, num_domains= configs.num_classes)
        self.classifier = common_network.feat_classifier(
            configs.num_classes, configs.final_out_channels // 4)

        self.seq_transformer2 = Seq_Transformer(patch_size=self.num_channels, dim=configs.TC.hidden_dim, depth=4, heads=4, mlp_dim=64)
        self.seq_transformer1 = Seq_Transformer(patch_size=self.num_channels, dim=configs.TC.hidden_dim, depth=4, heads=4, mlp_dim=64)
        # Attention mechanism for z
        self.query_s = nn.Parameter(torch.randn(1, configs.final_out_channels, 1))  # Query for s (1 x Ch x 1)
        self.query_c = nn.Parameter(torch.randn(1, configs.final_out_channels, 1))  # Query for c (1 x Ch x 1)
        self.softmax = nn.Softmax(dim=-1)
    
    def context(self, features_aug1):
        z_aug1 = features_aug1  # features are (batch_size, #channels, seq_len)
        seq_len = z_aug1.shape[2]
        z_aug1 = z_aug1.transpose(1, 2)
        forward_seq = z_aug1

        s_t = self.seq_transformer1(forward_seq)
        c_t= self.seq_transformer2(forward_seq)
        s_t_head = self.projection_head1(s_t)
        c_t_head = self.projection_head2(c_t)
        return c_t_head,s_t_head 
    def forward(self, features_aug1, features_aug2):
        z_aug1 = features_aug1  # features are (batch_size, #channels, seq_len)
        seq_len = z_aug1.shape[2]
        z_aug1 = z_aug1.transpose(1, 2)

        z_aug2 = features_aug2
        z_aug2 = z_aug2.transpose(1, 2)

        batch = z_aug1.shape[0]
        forward_seq = z_aug1

        nce = 0 


        s_t = self.seq_transformer1(forward_seq)
        c_t= self.seq_transformer2(forward_seq)

        head_pros = self.projection_head1(s_t)
        head_proc =  self.projection_head2(c_t)
        disc_input = Adver_network.ReverseLayerF.apply(
            head_pros, 1.0)        
        disc_out = self.discriminator(disc_input)
        pred_out = self.classifier(head_proc)


    
        return nce,  head_pros, head_proc, disc_out, pred_out, s_t, c_t

