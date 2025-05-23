import os
import sys
import torch
import argparse
import warnings
from datetime import datetime
from torch.utils.data import Dataset, DataLoader
from trainers.abstract_trainer import AbstractTrainer
from utils.denoising_diffusion_pytorch import Trainer1D_Train as Trainer1D, Unet1D_cond_train as Unet1D_cond, GaussianDiffusion1Dcond_train as GaussianDiffusion1Dcond


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
        return self.ori_x[idx], self.ori_y[idx], self.packed_data[idx], self.lengths[idx], self.sample_idx, idx


parser = argparse.ArgumentParser()
parser.add_argument('--seed', default=1, type=int)
parser.add_argument('--task_id', default=1, type=int)
parser.add_argument('--results_folder', default='./', type=str)
parser.add_argument('--device', default='cuda', type=str)
parser.add_argument('--run_id', default=0, type=int, help='Run index (for seed/reproducibility)')
parser.add_argument('--batch_size', default=64, type=int)
parser.add_argument('--dataset', default='emg', type=str)
args = parser.parse_args()

home_dir = os.getcwd()
args.home_path = home_dir
datainfo = AbstractTrainer(args)




testuser = {
    'seed': args.run_id,
    'name': f"{args.dataset}_task_{args.task_id}_seed_{args.run_id}",
    'dataset': args.dataset,
    'sde': os.path.join(os.getcwd(), 'intermediate_results/', f"{args.dataset}_task_{args.task_id}_seed_{args.run_id}-sde.pt"),
    'segment': os.path.join(os.getcwd(), 'intermediate_results/', f"{args.dataset}_task_{args.task_id}_seed_{args.run_id}-segment.pth"),
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

if shapex[1] % 64 != 0:
    shapex[1] = 64 - (shapex[1] % 64) + shapex[1]

model_our = Unet1D_cond(
    dim=64,
    guide_dim=shapex[2],
    dim_mults=(1, 2, 4, 8),
    channels=shapex[2],
    context_using=False
)

diffusion = GaussianDiffusion1Dcond(
    model_our,
    seq_length=shapex[1],
    timesteps=100,
    objective='pred_noise'
).to(device)



trainer = Trainer1D(
    None,
    diffusion,
    dataloader=source_loaders,
    train_batch_size=shapex[0],
    train_lr=2e-4,
    train_num_steps=10000,
    gradient_accumulate_every=2,
    ema_decay=0.995,
    amp=False,
    results_folder=args.results_folder
)
trainer.train(testuser)
