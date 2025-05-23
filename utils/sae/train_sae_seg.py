import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from einops import rearrange
from tslearn.metrics import SoftDTWLossPyTorch as SoftDTWLoss
from accelerate import Accelerator
from ema_pytorch import EMA
from tqdm.auto import tqdm
from datetime import datetime

class Trainer1D:
    def __init__(
        self,
        pgm,
        dataloader,
        train_batch_size=16,
        gradient_accumulate_every=1,
        train_lr=1e-4,
        train_num_steps=100000,
        ema_update_every=10,
        ema_decay=0.995,
        adam_betas=(0.9, 0.99),
        results_folder='./',
        max_grad_norm=1.0
    ):
        self.accelerator = Accelerator()
        self.pgm = pgm
        self.batch_size = train_batch_size
        self.gradient_accumulate_every = gradient_accumulate_every
        self.max_grad_norm = max_grad_norm
        self.train_num_steps = train_num_steps
        self.dataloader = self.accelerator.prepare(dataloader)
        self.opt = optim.Adam(self.pgm.parameters(), lr=train_lr, betas=adam_betas)
        self.ema = EMA(pgm, beta=ema_decay, update_every=ema_update_every).to(self.device)
        self.results_folder = results_folder
        os.makedirs(self.results_folder, exist_ok=True)
        self.step = 0
        self.pgm, self.opt = self.accelerator.prepare(self.pgm, self.opt)
    
    @property
    def device(self):
        return self.accelerator.device

    def save(self, milestone, name):
        if self.accelerator.is_local_main_process:
            torch.save(self.pgm.state_dict(), os.path.join(self.results_folder, f'{name}-checkpoint.pth'))

    def train(self, name):
        mse = nn.MSELoss()
        softdtw = SoftDTWLoss(gamma=0.1)
        with tqdm(initial=self.step, total=self.train_num_steps, disable=not self.accelerator.is_main_process) as pbar:
            while self.step < self.train_num_steps:
                total_loss = 0.0
                for _ in range(self.gradient_accumulate_every):
                    dataxy = next(iter(self.dataloader))
                    lengths, seg = dataxy[3].cpu(), dataxy[2]
                    seg = seg.to(dtype=torch.float32).to(self.device)
                    x_0_, z = self.pgm(seg, lengths)
                    mask_data = (torch.arange(seg.size(1)).unsqueeze(0) < lengths.unsqueeze(1)).int().to(self.device)
                    mask_pad = (1 - mask_data).to(self.device)
                    mask_data = mask_data.unsqueeze(2).repeat(1, 1, seg.shape[-1])
                    mask_pad = mask_pad.unsqueeze(2).repeat(1, 1, seg.shape[-1])
                    loss_sae = mse(seg * mask_data, x_0_ * mask_data) + mse(seg * mask_pad, x_0_ * mask_pad)
                    loss_sae += 0.01 * softdtw(seg, z).sum()
                    loss_sae = loss_sae / self.gradient_accumulate_every
                    total_loss += loss_sae.item()
                    self.accelerator.backward(loss_sae)
                self.accelerator.clip_grad_norm_(self.pgm.parameters(), self.max_grad_norm)
                self.opt.step()
                self.opt.zero_grad()
                self.step += 1
                if self.accelerator.is_main_process:
                    self.ema.update()
                pbar.set_description(f'Loss: {total_loss:.4f}')
                pbar.update(1)
                self.save(self.step, name)
        self.accelerator.print('Training complete')
