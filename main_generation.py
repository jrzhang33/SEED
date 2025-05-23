import os
import torch
import sys
import argparse
import warnings
import numpy as np
from torch.utils.data import Dataset, DataLoader
from utils import fix_randomness
from trainers.abstract_trainer import AbstractTrainer
from dataloader.dataloader import data_generator
from Diffusion_model.denoising_diffusion_pytorch import Unet1D_cond, GaussianDiffusion1Dcond
from utils.train_seed import train_seed
from Featurenet.config_files.distrb_condition import cond_set

warnings.filterwarnings("ignore")

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class CustomDataset(Dataset):
    def __init__(self, packed_data, lengths, ori_x, ori_y, sample_idx):
        self.packed_data = packed_data
        self.lengths = torch.tensor(lengths)
        self.ori_x = ori_x
        self.ori_y = ori_y
        self.sample_idx = sample_idx

    def __len__(self):
        return len(self.lengths)

    def __getitem__(self, idx):
        return self.ori_x[idx], self.ori_y[idx], self.packed_data[idx], self.lengths[idx], self.sample_idx[idx], idx


parser = argparse.ArgumentParser()


parser.add_argument('--seed', default=1, type=int)
parser.add_argument('--task_id', default=1, type=int)
parser.add_argument('--results_folder', default='./', type=str)
parser.add_argument('--device', default='cuda', type=str)
parser.add_argument('--run_id', default=0, type=int, help='Run index (for seed/reproducibility)')
parser.add_argument('--batch_size', default=64, type=int)
parser.add_argument('--dataset', default='emg', type=str)


args = parser.parse_args()




testuser = {
    'seed': args.run_id,
    'name': f"{args.dataset}_task_{args.task_id}_seed_{args.run_id}",
    'dataset': args.dataset,
    'segment': os.path.join(os.getcwd(), 'intermediate_results/', f"{args.dataset}_task_{args.task_id}_seed_{args.run_id}-segment.pth"),
    'sde': os.path.join(os.getcwd(), 'intermediate_results/', f"{args.dataset}_task_{args.task_id}_seed_{args.run_id}-sde.pt"),
    'diff': os.path.join(os.getcwd(), f"intermediate_results/", f"{args.dataset}_task_{args.task_id}_seed_{args.run_id}-diff.pt"),
    'newdata': os.path.join(os.getcwd(), f"intermediate_results/", f"{args.dataset}_task_{args.task_id}_seed_{args.run_id}-newdata.pt"),

}

loaded_dataset_dict = torch.load(testuser['segment'])
source_loaders = DataLoader(
    CustomDataset(
        loaded_dataset_dict['packed_data'],
        loaded_dataset_dict['lengths'],
        loaded_dataset_dict['ori_x'],
        loaded_dataset_dict['ori_y'],
        loaded_dataset_dict['sample_idx']
    ), batch_size=64, shuffle=True
)



for minibatch in source_loaders:
    batch_size = minibatch[0].shape[0]
    shapex = [minibatch[0].shape[0], minibatch[0].shape[2], minibatch[0].shape[1]]
    break

model_dim_diff = 64  
model_our = Unet1D_cond(
    dim=64,
    num_classes=model_dim_diff,
    dim_mults=(1, 2, 4, 8),
    channels=shapex[2],
    context_using=False
)
diffusion = GaussianDiffusion1Dcond(
    model_our,
    seq_length=shapex[1],
    timesteps=100,
    ddim_sampling_eta=0.95,
    objective='pred_noise'
).to(args.device)


diffusion.load_state_dict(torch.load(testuser['diff'])['model'])
train_seed(diffusion, args, source_loaders, None, None, testuser)
print(testuser['newdata'])
