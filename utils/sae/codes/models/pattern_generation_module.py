import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
torch.set_default_dtype(torch.float)
from tslearn.metrics import SoftDTWLossPyTorch as SoftDTWLoss
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
import sys

from models.scaling_autoencoder import *
from models.utils_generation import *
import sys

model_path = 'trained_models/'

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



#############################
# Pattern generation module #
#############################


class PatternGenerationModule(nn.Module):
  """
    Pattern generation module incorporating scaling AE and pattern-conditioned diffusion network (random noise schedule for diffusion network)
    Args
      sae: torch network, scaling AE network
      pcdm: torch network, pattern-conditioned diffusion network
      condition: bool, True for learning conditioned on patterns in pcdm
  """
  def __init__(self, sae,  fc, condition=True, device=None):
    super().__init__()
    self.device = device
    self.sae = sae.to(device)
    self.fc = fc.to(device)

    self.condition = condition

  def forward(self, x,lengths):
    x= x.to(self.device)
    batch_size = x.shape[0]
    # SAE encoder
    z, (z_hidden, z_cell) = self.sae.encoder(x, lengths) 
    # SAE decoder
    packed_z = pack_padded_sequence(z, lengths, batch_first=True, enforce_sorted=False)
    x_ = self.sae.decoder(packed_z)
    x_ = x_.squeeze(-1)
    z_out = z.reshape(x_.size()) 
    disc_in1 = z_out.reshape(z_out.shape[0],-1)




    return x_, z_out

  def generate(self, p, lengths):
    self.sae.eval()
    self.pcdm.eval()
    p = p.to(self.device)
    with torch.no_grad():
      p = p.unsqueeze(1)
      # Sample noise
      batch_size, n_channels, series_len = p.shape
      z_noisy = torch.randn_like(p).to(self.device)
      # PCDM denoising
      z_ = self.denoising_process(z_noisy, p,
                                  batch_size, n_channels, series_len).to(self.device)
      # SAE decoder
      if self.condition:
        z_ = z_ + p
      z_ = z_.reshape(batch_size, -1, 1)
      packed_z = pack_padded_sequence(z_, lengths, batch_first=True, enforce_sorted=False)
      x_ = self.sae.decoder(packed_z)
      x_ = x_.squeeze(-1)
    return x_, z_.reshape(x_.size())

  def denoising_process(self, z_noisy, p,
                        batch_size, n_channels, series_len):
    z_ = z_noisy
    for _, t in enumerate(list(range(self.n_steps))[::-1]):
      timestep = torch.full((batch_size,), t, dtype=torch.float32, device=self.device)
      e_theta = self.pcdm.backward(z_, timestep, p)
      alpha_t = self.pcdm.alphas[t]
      alpha_t_bar = self.pcdm.alpha_bars[t]
      z_ = (1 / alpha_t.sqrt()) * (z_ - (1 - alpha_t) / (1 - alpha_t_bar).sqrt() * e_theta)
      # Found it also works without the control of magnitude during the denoising process
      if t > 0:
        eta = torch.randn(batch_size, n_channels, series_len).to(self.device)
        beta_t = self.pcdm.betas[t]
        prev_alpha_t_bar = self.pcdm.alpha_bars[t-1] if t > 0 else self.pcdm.alphas[0]
        beta_tilda_t = ((1 - prev_alpha_t_bar)/(1 - alpha_t_bar)) * beta_t
        sigma_t = beta_tilda_t.sqrt()
        z_ += sigma_t * eta
    return z_
















