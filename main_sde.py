import os
import sys
import torch
import numpy as np
from datetime import datetime
from torch.utils.data import Dataset, DataLoader
import argparse
from utils.sde_models.trainer.trainer import Trainer
from utils.sde_models.TC import TC
from utils.sde_models.model import base_Model
from configs.sde_configs import emg_FEA_Configs as Configs
class CustomDataset(Dataset):
    def __init__(self, packed_data, lengths, ori_x, ori_y, sample_idx=None):
        self.packed_data = packed_data
        self.lengths = torch.tensor(lengths)
        self.ori_x = ori_x
        self.ori_y = ori_y
        self.sample_idx = sample_idx


    def __len__(self):
        return len(self.lengths)

    def __getitem__(self, idx):
        return self.ori_x[idx], self.ori_y[idx], self.packed_data[idx], self.lengths[idx], self.sample_idx[idx]

# Use GPU if available
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# ----------------- Argument Parser ------------------
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='emg', type=str, help='Dataset name')
parser.add_argument('--task_id', default=1, type=int, help='Task index for source/target domain')
parser.add_argument('--run_id', default=0, type=int, help='Run index (for seed/reproducibility)')
parser.add_argument('--segment_K', default=5, type=int, help='Number of segments per instance')
args = parser.parse_args()

# ----------------- Dataset & Config Initialization ------------------


testuser = {
    'seed': args.run_id,
    'name': f"{args.dataset}_task_{args.task_id}_seed_{args.run_id}",
    'dataset': args.dataset,
    'segment': os.path.join(os.getcwd(), 'intermediate_results/', f"{args.dataset}_task_{args.task_id}_seed_{args.run_id}-segment.pth"),
}

# Load preprocessed data (SDE training input)
testuser['conditioner_input'] = testuser['segment']
loaded_data = torch.load(testuser['conditioner_input'])


dataset = CustomDataset(loaded_data['packed_data'], loaded_data['lengths'],
                        loaded_data['ori_x'], loaded_data['ori_y'],
                        loaded_data['sample_idx'])
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# ----------------- Model Initialization ------------------
sample_batch = next(iter(dataloader))
batch_size = sample_batch[0].shape[0]
input_channels, input_length = sample_batch[0].shape[2], sample_batch[0].shape[1]
padded_length = input_length if input_length % 64 == 0 else input_length + 64 - (input_length % 64)

shape = [batch_size, padded_length, input_channels]
model_dim = shape[1] * shape[2]

# Backbone for encoder (used in TSC)

configs = Configs.Config()
configs.batch_size = batch_size
configs.final_out_channels = sample_batch[2].shape[-1]

model = base_Model(configs).to(device)
temporal_contr_model = TC(configs, device).to(device)

model_optimizer = torch.optim.Adam(model.parameters(), lr=configs.lr, betas=(configs.beta1, configs.beta2), weight_decay=3e-4)
temporal_optimizer = torch.optim.Adam(temporal_contr_model.parameters(), lr=configs.lr, betas=(configs.beta1, configs.beta2), weight_decay=3e-4)

log_dir = os.path.join('experiments_logs', f"{args.dataset}_seed_{args.run_id}")
os.makedirs(log_dir, exist_ok=True)
logger = _logger(os.path.join(log_dir, f"logs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"))

# ----------------- Train the SDE ------------------
Trainer(model, temporal_contr_model, model_optimizer, temporal_optimizer,
        dataloader, dataloader, dataloader, device, logger, configs,
        log_dir, training_mode="self_supervised", testuser=testuser)
