from torch import nn
import torch
import torch.nn.functional as F
import numpy as np
def generate_continuous_mask(B, T, n=5, l=0.1):
    res = torch.full((B, T), True, dtype=torch.bool)
    if isinstance(n, float):
        n = int(n * T)
    n = max(min(n, T // 2), 1)
    
    if isinstance(l, float):
        l = int(l * T)
    l = max(l, 1)
    
    for i in range(B):
        for _ in range(n):
            t = np.random.randint(T-l+1)
            res[i, t:t+l] = False
    return res

def generate_binomial_mask(B, T, p=0.7):
    return torch.from_numpy(np.random.binomial(1, p, size=(B, T))).to(torch.bool)

class base_Model(nn.Module):
    def __init__(self, configs):
        super(base_Model, self).__init__()

        self.conv_block1 = nn.Sequential(
            nn.Conv1d(configs.input_channels, 32, kernel_size=configs.kernel_size,
                      stride=configs.stride, bias=False, padding=(configs.kernel_size//2)),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
            nn.Dropout(configs.dropout)
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=8, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1)
        )

        self.conv_block3 = nn.Sequential(
            nn.Conv1d(64, configs.final_out_channels, kernel_size=8, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(configs.final_out_channels),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
        )

        model_output_dim = configs.features_len
        self.logits = nn.Linear(model_output_dim * configs.final_out_channels, configs.num_classes)

    def forward(self, x_in,mask =None):
        x_ = x_in.permute(0, 2, 1)  # B x T x C
        nan_mask = ~x_.isnan().any(axis=-1)
        x_[~nan_mask] = 0
        x = x_.transpose(1, 2)  # B xCh x T
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x) # B x T x Ch 
        #copy x
        x_1 = x.clone()
        x_2 = x.clone()


        # generate & apply mask
        if mask is None:
            mask = 'all_true'
            x_out = x.reshape(x.size(0), -1)
            try:
                y_predic = self.logits(x_out)
            except:
                y_predic = None
            return y_predic,x

        
        if mask == 'binomial':
            mask = generate_binomial_mask(x.size(0), x.size(1)).to(x.device)
            mask_ = generate_binomial_mask(x.size(0), x.size(1)).to(x.device)
            mask &= nan_mask
            mask_ &= nan_mask
            x_1[~mask] = 0 
            x_2[~mask_] = 0

        return  x_1,x_2  #B xT x C
    
