import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
torch.set_default_dtype(torch.float)



####################################
# Scaling Autoencoder (Scaling AE) #
####################################

class ScalingAE(nn.Module):
  """
    Scaling AutoEncoder
    Args
      input_dim: dimension of input segments (1 for univariate)
      hidden_dim: dimension of hidden representation
      output_dim: dimension of output segments (1 for univariate)
      custom_pad_length: maximum padded length of segments, used to unify the representation lengths over different batches (optional)
  """
  def __init__(self, input_dim, hidden_dim, output_dim, custom_pad_length=None, device=None):
    super().__init__()
    self.device = device
    self.encoder = ScalingEncoder(input_dim, hidden_dim, custom_pad_length).to(device)
    self.decoder = ScalingDecoder(hidden_dim, output_dim, custom_pad_length).to(device)

  def forward(self, x, lengths):
    x, lengths = x.to(self.device), lengths.to(self.device)
    packed_z, (hidden, cell) = self.encoder(x, lengths)
    x_ = self.decoder(packed_z, (hidden, cell))
    return packed_z, x_


class ScalingEncoder(nn.Module):
  """ Encoder in Scaling AE """
  def __init__(self, input_dim, hidden_dim, custom_pad_length=None):
    super().__init__()
    self.n_layers = 2
    self.custom_pad_length = custom_pad_length
    self.lstm = nn.LSTM(input_dim, hidden_dim, self.n_layers, batch_first=True)
    # self.gru = nn.GRU(input_dim, hidden_dim, self.n_layers, batch_first=True)
    self.fc = nn.Linear(hidden_dim, hidden_dim)

  def forward(self, x, lengths):
    x = x.transpose(2,1) #batch.len,dim
    packed_x = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
    packed_x = packed_x.to(dtype=torch.float32)  # 修改 dtype 并重新赋值
    packed_z, (hidden, cell) = self.lstm(packed_x)
    z, _ = pad_packed_sequence(packed_z, batch_first=True, total_length=self.custom_pad_length)
    z = self.fc(z)
    return z, (hidden, cell)


class ScalingDecoder(nn.Module):
  """ Decoder in Scaling AE """
  def __init__(self, hidden_dim, output_dim, custom_pad_length=None):
    super().__init__()
    self.n_layers = 2
    self.custom_pad_length = custom_pad_length
    self.lstm = nn.LSTM(hidden_dim, hidden_dim, self.n_layers, batch_first=True)
    # self.gru = nn.GRU(hidden_dim, hidden_dim, self.n_layers, batch_first=True)
    self.fc = nn.Linear(hidden_dim, output_dim)

  def forward(self, packed_z, hidden=None, cell=None):
    # packed_x_, _ = self.lstm(packed_z, (hidden, cell))
    # return packed_x_
    packed_x_, _ = self.lstm(packed_z)
    x_, _ = pad_packed_sequence(packed_x_, batch_first=True, total_length = self.custom_pad_length)
    x_ = self.fc(x_)
    return x_









