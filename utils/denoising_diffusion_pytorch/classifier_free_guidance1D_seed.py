import math
import copy
from pathlib import Path
from random import random
from functools import partial
from collections import namedtuple
from multiprocessing import cpu_count

import torch
from torch import nn, einsum
import torch.nn.functional as F
from torch.cuda.amp import autocast

from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange

from tqdm.auto import tqdm
from Featurenet.utils.util import set_random_seed
set_random_seed(43)
# constants

ModelPrediction =  namedtuple('ModelPrediction', ['pred_noise', 'pred_x_start'])
import os
import matplotlib.pyplot as plt
import numpy as np
# helpers functions
# def plotdiff(rep, times):
#     batch_size, channels, length = rep.size()
#     folder_name = './stept1'
#     os.makedirs(folder_name, exist_ok=True)
#     # Visualize rep
#     for i in range(2):
#         for j in range(1):
#             # Get single channel data
#             channel_data = rep[i, j, :]

#             # Create plot
#             plt.plot(channel_data.cpu())
#             plt.title(f'rep Channel {j} Batch {i} Step_{times}')

#             # Save plot as PNG
#             file_name = f'rep_channel_{j}_batch_{i}_Step_{times}.png'
#             plt.savefig(os.path.join(folder_name, file_name))


#             # Clear current plot
#             plt.clf()
#             channel_data_np = channel_data.cpu().numpy()

#             channel_data_fft = np.fft.fft(channel_data_np)

#             # Calculate frequency values
#             freq = np.fft.fftfreq(len(channel_data_np))

#             # Create plot for frequency spectrum
#             plt.plot(freq, np.abs(channel_data_fft))
#             plt.title(f'rep Channel {j} Batch {i} Step_{times} Frequency Spectrum')
#             positive_freq = freq[:len(freq)//2]  # Consider only positive frequencies
#             positive_spectrum = np.abs(channel_data_fft)[:len(freq)//2]  # Magnitudes of positive frequencies

#             # Calculate the centroid
#             centroid = np.sum(positive_freq * positive_spectrum) / np.sum(positive_spectrum)

#             # Calculate the energy
#             energy = np.sum(np.square(positive_spectrum))
#             print("Rep: ")    
#             print("REP:Centroid:", centroid)
#             print("REP: Energy:", energy)
#             # Save plot as PNG in the folder
#             file_name = f'rep_channel_{j}_batch_{i}_Step_{times}_freq.png'
#             plt.savefig(os.path.join(folder_name, file_name))
#             plt.clf()


def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

def identity(t, *args, **kwargs):
    return t

def cycle(dl):
    while True:
        for data in dl:
            yield data

def has_int_squareroot(num):
    return (math.sqrt(num) ** 2) == num

def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr

def convert_image_to_fn(img_type, image):
    if image.mode != img_type:
        return image.convert(img_type)
    return image

# normalization functions

def normalize_to_neg_one_to_one(img):
    return img * 2 - 1

def unnormalize_to_zero_to_one(t):
    return (t + 1) * 0.5

# classifier free guidance functions

def uniform(shape, device):
    return torch.zeros(shape, device = device).float().uniform_(0, 1)

def prob_mask_like(shape, prob, device):
    if prob == 1:
        return torch.ones(shape, device = device, dtype = torch.bool)
    elif prob == 0:
        return torch.zeros(shape, device = device, dtype = torch.bool)
    else:
        return torch.zeros(shape, device = device).float().uniform_(0, 1) < prob

# small helper modules

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x

def Upsample(dim, dim_out = None):
    return nn.Sequential(
        nn.Upsample(scale_factor = 2, mode = 'nearest'),
        nn.Conv1d(dim, default(dim_out, dim), 3, padding = 1)
    )

def Downsample(dim, dim_out = None):
    return nn.Conv1d(dim, default(dim_out, dim), 4, 2, 1)

class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.g = nn.Parameter(torch.ones(1, dim, 1))

    def forward(self, x):
        return F.normalize(x, dim = 1) * self.g * (x.shape[1] ** 0.5)

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = RMSNorm(dim)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)

# sinusoidal positional embeds

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class RandomOrLearnedSinusoidalPosEmb(nn.Module):
    """ following @crowsonkb 's lead with random (learned optional) sinusoidal pos emb """
    """ https://github.com/crowsonkb/v-diffusion-jax/blob/master/diffusion/models/danbooru_128.py#L8 """

    def __init__(self, dim, is_random = False):
        super().__init__()
        assert (dim % 2) == 0
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim), requires_grad = not is_random)

    def forward(self, x):
        x = rearrange(x, 'b -> b 1')
        freqs = x * rearrange(self.weights, 'd -> 1 d') * 2 * math.pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim = -1)
        fouriered = torch.cat((x, fouriered), dim = -1)
        return fouriered

# building block modules

class Block(nn.Module):
    def __init__(self, dim, dim_out, groups = 8):
        super().__init__()
        self.proj = nn.Conv1d(dim, dim_out, 3, padding = 1)
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

    def forward(self, x, scale_shift = None):
        x = self.proj(x)
        x = self.norm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)
        return x

class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, *, time_emb_dim = None, classes_emb_dim = None, groups = 8):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(int(time_emb_dim) + int(classes_emb_dim), dim_out * 2)
        ) if exists(time_emb_dim) or exists(classes_emb_dim) else None

        self.block1 = Block(dim, dim_out, groups = groups)
        self.block2 = Block(dim_out, dim_out, groups = groups)
        self.res_conv = nn.Conv1d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb = None, class_emb = None):

        scale_shift = None
        if exists(self.mlp) and (exists(time_emb) or exists(class_emb)):
            if time_emb.shape[0]!= class_emb.shape[0]:
                class_emb = class_emb[:time_emb.shape[0]]
            cond_emb = tuple(filter(exists, (time_emb, class_emb)))
            cond_emb = torch.cat(cond_emb, dim = -1)
            cond_emb = self.mlp(cond_emb)
            cond_emb = rearrange(cond_emb, 'b c -> b c 1')
            scale_shift = cond_emb.chunk(2, dim = 1)
            
        h = self.block1(x, scale_shift = scale_shift)

        h = self.block2(h)

        return h + self.res_conv(x)

class LinearAttention(nn.Module):
    def __init__(self, dim, heads = 4, dim_head = 32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv1d(dim, hidden_dim * 3, 1, bias = False)

        self.to_out = nn.Sequential(
            nn.Conv1d(hidden_dim, dim, 1),
            RMSNorm(dim)
        )

    def forward(self, x):
        b, c, n = x.shape
        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) n -> b h c n', h = self.heads), qkv)

        q = q.softmax(dim = -2)
        k = k.softmax(dim = -1)

        q = q * self.scale        

        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)

        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
        out = rearrange(out, 'b h c n -> b (h c) n', h = self.heads)
        return self.to_out(out)

class Attention(nn.Module):
    def __init__(self, dim, heads = 4, dim_head = 32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads

        self.to_qkv = nn.Conv1d(dim, hidden_dim * 3, 1, bias = False)
        self.to_out = nn.Conv1d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, n = x.shape
        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) n -> b h c n', h = self.heads), qkv)

        q = q * self.scale

        sim = einsum('b h d i, b h d j -> b h i j', q, k)
        attn = sim.softmax(dim = -1)
        out = einsum('b h i j, b h d j -> b h i d', attn, v)

        out = rearrange(out, 'b h n d -> b (h d) n')
        return self.to_out(out)

# model

class Unet1D_cond(nn.Module):
    def __init__(
        self,
        dim,
        num_classes,
        cond_drop_prob = 0.5,
        init_dim = None,
        out_dim = None,
        dim_mults=(1, 2, 4, 8),
        channels = 3,
        resnet_block_groups = 8,
        learned_variance = False,
        learned_sinusoidal_cond = False,
        random_fourier_features = False,
        learned_sinusoidal_dim = 16,
        attn_dim_head = 32,
        attn_heads = 4,
        context_using = False,
    ):
        super().__init__()

        # classifier free guidance stuff

        self.cond_drop_prob = cond_drop_prob

        # determine dimensions

        self.channels = channels
        input_channels = channels

        init_dim = default(init_dim, dim)
        self.init_conv = nn.Conv1d(input_channels, init_dim, 7, padding = 3)

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        block_klass = partial(ResnetBlock, groups = resnet_block_groups)

        # time embeddings

        time_dim = dim * 4

        self.random_or_learned_sinusoidal_cond = learned_sinusoidal_cond or random_fourier_features

        if self.random_or_learned_sinusoidal_cond:
            sinu_pos_emb = RandomOrLearnedSinusoidalPosEmb(learned_sinusoidal_dim, random_fourier_features)
            fourier_dim = learned_sinusoidal_dim + 1
        else:
            sinu_pos_emb = SinusoidalPosEmb(dim)
            fourier_dim = dim

        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )

        # style embeddings

        self.classes_emb = nn.Linear(num_classes[0], dim)
        self.classes_emb_label = nn.Embedding(num_classes[1], dim)
  
        self.null_classes_emb = nn.Parameter(torch.zeros(dim))

        classes_dim = dim * 4

        self.classes_mlp = nn.Sequential(
            nn.Linear(dim, classes_dim),
            nn.GELU(),
            nn.Linear(classes_dim, classes_dim)
        )

        # layers

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(nn.ModuleList([
                block_klass(dim_in, dim_in, time_emb_dim = time_dim, classes_emb_dim = classes_dim),
                block_klass(dim_in, dim_in, time_emb_dim = time_dim, classes_emb_dim = classes_dim),
                Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                Downsample(dim_in, dim_out) if not is_last else nn.Conv1d(dim_in, dim_out, 3, padding = 1)
            ]))

        mid_dim = dims[-1]
        self.mid_block1 = block_klass(mid_dim, mid_dim, time_emb_dim = time_dim, classes_emb_dim = classes_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, Attention(mid_dim, dim_head = attn_dim_head, heads = attn_heads)))
        self.mid_block2 = block_klass(mid_dim, mid_dim, time_emb_dim = time_dim, classes_emb_dim = classes_dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind == (len(in_out) - 1)

            self.ups.append(nn.ModuleList([
                block_klass(dim_out + dim_in, dim_out, time_emb_dim = time_dim, classes_emb_dim = classes_dim),
                block_klass(dim_out + dim_in, dim_out, time_emb_dim = time_dim, classes_emb_dim = classes_dim),
                Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                Upsample(dim_out, dim_in) if not is_last else  nn.Conv1d(dim_out, dim_in, 3, padding = 1)
            ]))

        default_out_dim = channels * (1 if not learned_variance else 2)
        self.out_dim = default(out_dim, default_out_dim)

        self.final_res_block = block_klass(dim * 2, dim, time_emb_dim = time_dim, classes_emb_dim = classes_dim)
        self.final_conv = nn.Conv1d(dim, self.out_dim, 1)

    def forward_with_cond_scale(  
        self,
        *args,
        
        cond_scale = 1.,
        rescaled_phi = 0.,
        **kwargs
    ):
        logits = self.forward(*args, cond_drop_prob = 0., **kwargs)
        


        if cond_scale == 1:
            return logits

        null_logits = self.forward(*args, cond_drop_prob = 1., **kwargs)
        right_res = (logits - null_logits.clone()) * cond_scale
   # args: x, t, classes,  combination_dict, classes_gen, combination_dict_gen, aug_spe, aug_gen, labels,
        for i in range(len(args[3])):
            if len(args[3]) == 0:
                null_logits[i] = null_logits[i] /  len(args[5][i])
            elif len(args[5]) == 0:
                null_logits[i] = null_logits[i] / len(args[3][i])
            else:
                null_logits[i] = null_logits[i] / (len(args[3][i]) + len(args[5][i]))
            # print("each cond num:", (len(args[3][i]) + len(args[5][i])))

        scaled_logits = null_logits + right_res
       
        if rescaled_phi == 0.:
            return scaled_logits

        std_fn = partial(torch.std, dim = tuple(range(1, scaled_logits.ndim)), keepdim = True)
        if rescaled_phi != 0:
            rescaled_logits = scaled_logits * (std_fn(logits) / std_fn(scaled_logits))

        return rescaled_logits * rescaled_phi + scaled_logits * (1. - rescaled_phi)

    def forward_feature(
        self,
        x,
        time,
        classes,
        cond_drop_prob = None
    ):
        input = x.clone()
        batch, device = x.shape[0], x.device

        cond_drop_prob = default(cond_drop_prob, self.cond_drop_prob)

        # derive condition, with condition dropout for classifier free guidance        
        try:
            classes_emb = self.classes_emb(classes)
        except:
            classes_emb = self.classes_emb_label(classes)

        feature_vectors = []  # Store feature vectors from each block

        if cond_drop_prob > 0:
            keep_mask = prob_mask_like((batch,), 1 - cond_drop_prob, device = device)
            null_classes_emb = repeat(self.null_classes_emb, 'd -> b d', b = batch)

            classes_emb = torch.where(
                rearrange(keep_mask, 'b -> b 1'),
                classes_emb,
                null_classes_emb
            )

        c = self.classes_mlp(classes_emb)

        # unet

        x = self.init_conv(x)
        r = x.clone()

        t = self.time_mlp(time)

        h = []

        for block1, block2, attn, downsample in self.downs:
            x = block1(x, t, c)
            h.append(x)

            x = block2(x, t, c)
            x = attn(x)
            h.append(x)

            x = downsample(x)
            feature_vectors.append(x)

        x = self.mid_block1(x, t, c)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t, c)

        for block1, block2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim = 1)
            x = block1(x, t, c)

            x = torch.cat((x, h.pop()), dim = 1)
            x = block2(x, t, c)
            x = attn(x)

            x = upsample(x)
            feature_vectors.append(x)

        x = torch.cat((x, r), dim = 1)

        x = self.final_res_block(x, t, c)
       #####
        max_c = max(tensor.size(1) for tensor in feature_vectors)
        max_len = max(tensor.size(2) for tensor in feature_vectors)

        # Reshape tensors to have a consistent size
        resized_tensors = []
        for tensor in feature_vectors:
            resized_tensor = torch.zeros(tensor.size(0), max_c, max_len)
            resized_tensor[:, :tensor.size(1), :tensor.size(2)] = tensor
            resized_tensors.append(resized_tensor)

        # Concatenate the resized tensors along the second dimension
        concatenated_features = torch.cat(resized_tensors, dim=1) #32,4096,128
        #####
     #   combined_features = F.interpolate(concatenated_features, size=input.size()[2:], mode='bilinear', align_corners=False)
        return  concatenated_features



    def forward(self, x_input, time_input, classesbatch, combination_dict, classesbatch_gen, combination_dict_gen, aug_spe, aug_gen, labels, cond_drop_prob=None):
        if combination_dict == None:
            return self.forward_onecond(x_input, time_input, classesbatch, cond_drop_prob)
        # Preprocess all inputs
        all_inputs = []
        if len(classesbatch) != 0:
            for i in range(len(combination_dict)):
                for j in range(len(combination_dict[i])):
                    classes = classesbatch[combination_dict[i][j]].unsqueeze(0)
                    x = x_input[i].unsqueeze(0)
                    time = time_input[i].unsqueeze(0)
                    all_inputs.append((x, time, classes))
        len_comb = len(all_inputs)
        
        
        if len(classesbatch_gen) != 0:
            for i in range(len(combination_dict_gen)):
                for j in range(len(combination_dict_gen[i])):
                    classes = classesbatch_gen[combination_dict_gen[i][j]].unsqueeze(0)
                    x = x_input[i].unsqueeze(0)
                    time = time_input[i].unsqueeze(0)
                    all_inputs.append((x, time, classes))
        len_comb_label = len(all_inputs)
        if len(aug_spe) != 0:
            for i in range(len(combination_dict)):
                for j in range(len(combination_dict[i])):
                    classes = aug_spe[combination_dict[i][j]].unsqueeze(0)
                    x = x_input[i].unsqueeze(0)
                    time = time_input[i].unsqueeze(0)
                    all_inputs.append((x, time, classes))
        len_comb_aug = len(all_inputs)
        if len(aug_gen) != 0:
            for i in range(len(combination_dict_gen)):
                for j in range(len(combination_dict_gen[i])):
                    classes = aug_gen[combination_dict_gen[i][j]].unsqueeze(0)
                    x = x_input[i].unsqueeze(0)
                    time = time_input[i].unsqueeze(0)
                    all_inputs.append((x, time, classes))
        len_comb_aug_spe = len(all_inputs)

        # if len(classesbatch) != 0:
        #     for i in range(len(combination_dict)):
        #         for j in range(len(combination_dict[i])):
        #             classes = labels[combination_dict[i][j]].unsqueeze(0)
        #             x = x_input[i].unsqueeze(0)
        #             time = time_input[i].unsqueeze(0)
        #             all_inputs.append((x, time, classes))
        #             break
        # Convert list of tuples to separate lists
        all_x, all_time, all_classes = zip(*all_inputs)

        # Convert lists to tensors
        all_x = torch.stack(all_x).cuda()
        all_time = torch.stack(all_time).cuda()
      #  all_classes = torch.stack(all_classes).cuda()
        if len(classesbatch) != 0:
            try:
                classes_emb = self.classes_emb(torch.stack(all_classes[:len_comb]).cuda())
            except:
                classes_emb = self.classes_emb_label(torch.stack(all_classes[:len_comb]).cuda())
        else:
            classes_emb = []
        if len_comb != len(all_classes):
            try:
                classes_emb_label = self.classes_emb_label( torch.stack(all_classes[len_comb:]).cuda())
            except:
                classes_emb_label = self.classes_emb(torch.stack(all_classes[len_comb:]).cuda())
        
            
        if len_comb == len(all_classes) and len_comb != 0:
            all_classes = classes_emb.squeeze()
        elif len_comb == 0:
            all_classes = classes_emb_label.squeeze()
        elif len(classes_emb) != 0 and len(classes_emb_label) != 0:
            all_classes = torch.cat((classes_emb, classes_emb_label), dim=0).squeeze()
    
        # all_classes = torch.stack(all_classes).cuda()
        

        # Rest of your code here, but replace x, time, classes with all_x, all_time, all_classes
        # ...
        x, time, classes = all_x.squeeze(), all_time.squeeze(), all_classes.squeeze()
        batch, device = x.shape[0], x.device

        cond_drop_prob = default(cond_drop_prob, self.cond_drop_prob)

        # derive condition, with condition dropout for classifier free guidance        
  
        # classes_emb = self.classes_emb(classes[:len_comb])
        # classes_emb_label = self.classes_emb_label(classes[len_comb:])
 
        # if len(labels) != 0:
        #     classes_emb_label = self.classes_emb_label(torch.tensor(labels).cuda())
        #     classes_emb = torch.concatenate([classes_emb, classes_emb_label],axis = 0)
        batch = classes.shape[0]


        if cond_drop_prob > 0:
            keep_mask = prob_mask_like((batch,), 1 - cond_drop_prob, device = device)
            null_classes_emb = repeat(self.null_classes_emb, 'd -> b d', b = batch)
            # if classes_emb.shape[0] != null_classes_emb.shape[0]:
            #     class_emb = class_emb[:null_classes_emb.shape[0]]

            classes = torch.where(
                rearrange(keep_mask, 'b -> b 1'),
                classes,
                null_classes_emb
            )

        c = self.classes_mlp(classes)

        # unet

        x = self.init_conv(x)
        r = x.clone()

        t = self.time_mlp(time)

        h = []

        for block1, block2, attn, downsample in self.downs:
            x = block1(x, t, c)
            h.append(x)

            x = block2(x, t, c)
            x = attn(x)
            h.append(x)

            x = downsample(x)


        x = self.mid_block1(x, t, c)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t, c)

        for block1, block2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim = 1)
            x = block1(x, t, c)

            x = torch.cat((x, h.pop()), dim = 1)
            x = block2(x, t, c)
            x = attn(x)

            x = upsample(x)
            

        x = torch.cat((x, r), dim = 1)

        x = self.final_res_block(x, t, c)

        all_logits = self.final_conv(x)
        
        # Collect outputs and sum them up
        
        logits = torch.zeros(x_input.shape).cuda()
        if len(classesbatch) != 0:
            start = 0
            k =  len_comb_aug_spe
            for i in range(len(combination_dict)):
                for j in range(len(combination_dict[i])):
                    all_logits[start+j] = all_logits[start+j] / len(combination_dict[i]) 
                    one_logits = all_logits[start+j]
             
                    logits[i] += one_logits 
                    if cond_drop_prob == 1.0: assert (one_logits == all_logits[start]).all()
                start += len(combination_dict[i])
                k = k+1
        # if len(aug_spe) != 0:
        #     start = len_comb
        #     # logits_gen = torch.zeros(x_input.shape).cuda()
        #     k =  len_comb_aug_spe
        #     for i in range(len(combination_dict)):
        #         for j in range(len(combination_dict[i])):
        #             one_logits = all_logits[start+j] 
        #             logits[i]  += one_logits
        #           #  logits_gen[i] += one_logits
        #             if cond_drop_prob == 1.0: assert (one_logits == all_logits[start]).all()

        #         start += len(combination_dict[i])
        #         k = k+1
            
        if len(classesbatch_gen) != 0:
            start = len_comb
            # logits_gen = torch.zeros(x_input.shape).cuda()
            k =  len_comb_aug_spe
            for i in range(len(combination_dict_gen)):
                for j in range(len(combination_dict_gen[i])):
                    all_logits[start+j] = all_logits[start+j] / len(combination_dict_gen[i]) 
                    one_logits = all_logits[start+j] 
                    logits[i]  += one_logits 
                    # logits_gen[i] += one_logits
                    if cond_drop_prob == 1.0: assert (one_logits == all_logits[start]).all()
                start += len(combination_dict_gen[i])
                k = k+1 
        if (len(aug_gen) != 0 and len(aug_spe) != 0):   
            assert len(classesbatch_gen) != len(aug_spe) 

 
        # for i in range(len(logits)):
        #     if len(classesbatch) != 0:
        #         logits[i] = logits[i]
        #     if len(aug_spe) != 0:
        #         logits[i] += logits_gen[i] 
        #     if len(classesbatch_gen) != 0:
        #         logits[i] += logits_gen[i] 
        assert start == len(all_logits)



    ####multi condition    
        # if len(classesbatch) != 0:
            
        #     start = 0
        #     k =  len_comb_aug_spe
        #     for i in range(len(combination_dict)):
        #         for j in range(len(combination_dict[i])):
        #             one_logits = all_logits[start+j]
        #             logits[i] += one_logits
        #             if cond_drop_prob == 1.0: assert (one_logits == all_logits[start]).all()
        #         start += len(combination_dict[i])
        #         k = k+1
        # if len(classesbatch_gen) != 0:
        #     start = len_comb
        #     logits_gen = torch.zeros(x_input.shape).cuda()
        #     k =  len_comb_aug_spe
        #     for i in range(len(combination_dict_gen)):
        #         for j in range(len(combination_dict_gen[i])):
        #             one_logits = all_logits[start+j]
        #             logits_gen[i] += one_logits
        #             if cond_drop_prob == 1.0: assert (one_logits == all_logits[start]).all()
        #         if len(labels) != 0:
        #             logits_gen[i] += all_logits[k]
        #         start += len(combination_dict_gen[i])
        #         k = k+1
        #     assert len(logits) == len(logits_gen)
        # if len(aug_spe) != 0:
        #     start = len_comb_label
        #     logits_spe_aug = torch.zeros(x_input.shape).cuda()
        #     k = len_comb_aug_spe
        #     for i in range(len(combination_dict)):
        #         for j in range(len(combination_dict[i])):
        #             one_logits = all_logits[start+j]
        #             logits_spe_aug[i] += one_logits
        #             if cond_drop_prob == 1.0: assert (one_logits == all_logits[start]).all()
        #         start += len(combination_dict[i])
        #         k = k+1
        #     assert len(logits) == len(logits_spe_aug) 
        # if len(aug_gen) != 0:
        #     start = len_comb_aug
        #     logits_gen_aug = torch.zeros(x_input.shape).cuda()
        #     k =  len_comb_aug_spe
        #     for i in range(len(combination_dict_gen)):
        #         for j in range(len(combination_dict_gen[i])):
        #             one_logits = all_logits[start+j]
        #             logits_gen_aug[i] += one_logits
        #             if cond_drop_prob == 1.0: assert (one_logits == all_logits[start]).all()

        #         start += len(combination_dict_gen[i])
        #         k = k+1
        #     assert len(logits) == len(logits_gen_aug) 
    #     for i in range(len(logits)):
    #         if len(classesbatch) != 0:
    #             logits[i] = logits[i]
    #         if len(classesbatch_gen) != 0:
    #             logits[i] += logits_gen[i] 
    #         if len(aug_spe) != 0:
    #              logits[i] += logits_spe_aug[i]
    #         if len(aug_gen) != 0:
    #              logits[i] +=  logits_gen_aug[i]
    #  #   assert start == len(all_logits)
    #     assert start + len(labels) == len(all_logits)

####multi condition   end

        return logits




    def forward_onecond(
        self,
        x,
        time,
        classes,
        cond_drop_prob = None
    ):
        batch, device = x.shape[0], x.device

        cond_drop_prob = default(cond_drop_prob, self.cond_drop_prob)

        # derive condition, with condition dropout for classifier free guidance        
        try:
            classes_emb = self.classes_emb(classes)
        except:
            classes_emb = self.classes_emb(classes)

        if cond_drop_prob > 0:
            keep_mask = prob_mask_like((batch,), 1 - cond_drop_prob, device = device)
            null_classes_emb = repeat(self.null_classes_emb, 'd -> b d', b = batch)

            classes_emb = torch.where(
                rearrange(keep_mask, 'b -> b 1'),
                classes_emb,
                null_classes_emb
            )

        c = self.classes_mlp(classes_emb)

        # unet

        x = self.init_conv(x)
        r = x.clone()

        t = self.time_mlp(time)

        h = []

        for block1, block2, attn, downsample in self.downs:
            x = block1(x, t, c)
            h.append(x)

            x = block2(x, t, c)
            x = attn(x)
            h.append(x)

            x = downsample(x)

        x = self.mid_block1(x, t, c)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t, c)

        for block1, block2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim = 1)
            x = block1(x, t, c)

            x = torch.cat((x, h.pop()), dim = 1)
            x = block2(x, t, c)
            x = attn(x)

            x = upsample(x)

        x = torch.cat((x, r), dim = 1)

        x = self.final_res_block(x, t, c)
        return self.final_conv(x)
# gaussian diffusion trainer class

def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def linear_beta_schedule(timesteps):
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype = torch.float64)

def cosine_beta_schedule(timesteps, s = 0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype = torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)

class GaussianDiffusion1Dcond(nn.Module):
    def __init__(
        self,
        model,
        *,
        seq_length,
        timesteps = 1000,
        sampling_timesteps = None,
        objective = 'pred_noise',
        beta_schedule = 'cosine',
        ddim_sampling_eta = 1.,
        offset_noise_strength = 0.,
        min_snr_loss_weight = False,
        min_snr_gamma = 5
    ):
        super().__init__()
        assert not (type(self) == GaussianDiffusion1Dcond and model.channels != model.out_dim)
        assert not model.random_or_learned_sinusoidal_cond

        self.model = model
        self.channels = self.model.channels

        self.seq_length = seq_length

        self.objective = objective

        assert objective in {'pred_noise', 'pred_x0', 'pred_v'}, 'objective must be either pred_noise (predict noise) or pred_x0 (predict image start) or pred_v (predict v [v-parameterization as defined in appendix D of progressive distillation paper, used in imagen-video successfully])'

        if beta_schedule == 'linear':
            betas = linear_beta_schedule(timesteps)
        elif beta_schedule == 'cosine':
            betas = cosine_beta_schedule(timesteps)
        else:
            raise ValueError(f'unknown beta schedule {beta_schedule}')

        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value = 1.)

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)

        # sampling related parameters

        self.sampling_timesteps = default(sampling_timesteps, timesteps) # default num sampling timesteps to number of timesteps at training

        assert self.sampling_timesteps <= timesteps
        self.is_ddim_sampling = self.sampling_timesteps < timesteps
        self.ddim_sampling_eta = ddim_sampling_eta

        # helper function to register buffer from float64 to float32

        register_buffer = lambda name, val: self.register_buffer(name, val.to(torch.float32))

        register_buffer('betas', betas)
        register_buffer('alphas_cumprod', alphas_cumprod)
        register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others

        register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)

        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)

        register_buffer('posterior_variance', posterior_variance)

        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain

        register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min =1e-20)))
        register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        register_buffer('posterior_mean_coef2', (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))

        # offset noise strength - 0.1 was claimed ideal

        self.offset_noise_strength = offset_noise_strength

        # loss weight

        snr = alphas_cumprod / (1 - alphas_cumprod)

        maybe_clipped_snr = snr.clone()
        if min_snr_loss_weight:
            maybe_clipped_snr.clamp_(max = min_snr_gamma)

        if objective == 'pred_noise':
            loss_weight = maybe_clipped_snr / snr
        elif objective == 'pred_x0':
            loss_weight = maybe_clipped_snr
        elif objective == 'pred_v':
            loss_weight = maybe_clipped_snr / (snr + 1)

        register_buffer('loss_weight', loss_weight)

    @property
    def device(self):
        return self.betas.device

    def predict_start_from_noise(self, x_t, t, noise):
        if t.shape[0] != noise.shape[0]:
            t = t[:noise.shape[0]]
            x_t = x_t[:noise.shape[0]]

        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def predict_noise_from_start(self, x_t, t, x0):
        return (
            (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) / \
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )

    def predict_v(self, x_start, t, noise):
        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * noise -
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * x_start
        )

    def predict_start_from_v(self, x_t, t, v):
        return (
            extract(self.sqrt_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape) * v
        )

    def q_posterior(self, x_start, x_t, t):
        if x_start.shape[0] != t.shape[0]:
            t = t[:x_start.shape[0]]
            x_t = x_t[:x_start.shape[0]]

        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def model_predictions(self, x, t, classes,  combination_dict, classes_gen,  combination_dict_gen, aug_spe, aug_gen, labels, cond_scale = 6., rescaled_phi = 0.7, clip_x_start = False):
        model_output = self.model.forward_with_cond_scale(x, t, classes,  combination_dict, classes_gen, combination_dict_gen, aug_spe, aug_gen, labels, cond_scale = cond_scale, rescaled_phi = rescaled_phi)
        maybe_clip = partial(torch.clamp, min = -1., max = 1.) if clip_x_start else identity

        if self.objective == 'pred_noise':
            pred_noise = model_output
            x_start = self.predict_start_from_noise(x, t, pred_noise)
            x_start = maybe_clip(x_start)

        elif self.objective == 'pred_x0':
            x_start = model_output
            x_start = maybe_clip(x_start)
            pred_noise = self.predict_noise_from_start(x, t, x_start)

        elif self.objective == 'pred_v':
            v = model_output
            x_start = self.predict_start_from_v(x, t, v)
            x_start = maybe_clip(x_start)
            pred_noise = self.predict_noise_from_start(x, t, x_start)

        return ModelPrediction(pred_noise, x_start)

    def p_mean_variance(self, x, t, classes,  combination_dict, classes_gen, combination_dict_gen,aug_spe, aug_gen,labels, cond_scale, rescaled_phi, clip_denoised = True):
        preds = self.model_predictions(x, t, classes, combination_dict, classes_gen, combination_dict_gen,aug_spe, aug_gen, labels, cond_scale, rescaled_phi)
        x_start = preds.pred_x_start

        if clip_denoised:
            x_start.clamp_(-1., 1.)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start = x_start, x_t = x, t = t)
        return model_mean, posterior_variance, posterior_log_variance, x_start

    @torch.no_grad()
    def p_sample(self, x, t: int, classes, combination_dict, classes_gen, combination_dict_gen,  aug_spe, aug_gen,keys_tensor, cond_scale = 6., rescaled_phi = 0.7, clip_denoised = True, ):
        b, *_, device = *x.shape, x.device
        batched_times = torch.full((x.shape[0],), t, device = x.device, dtype = torch.long)
        model_mean, _, model_log_variance, x_start = self.p_mean_variance(x = x, t = batched_times, classes = classes,  combination_dict =  combination_dict, classes_gen =  classes_gen, combination_dict_gen = combination_dict_gen, aug_spe =  aug_spe, aug_gen = aug_gen, labels = keys_tensor, cond_scale = cond_scale, rescaled_phi = rescaled_phi, clip_denoised = clip_denoised)
        ####### 1.2
        noise = (torch.randn_like(x) * 1.2) if t > 0 else 0.
        if t!=0:
            if model_mean.shape[0] != noise.shape[0]:
                noise = noise[:model_mean.shape[0]]
        pred_img = model_mean + (0.5 * model_log_variance).exp() * noise
        return pred_img, x_start
    



    @torch.no_grad()
    def p_sample_loop(self, classes, combination_dict, classes_gen, combination_dict_gen, aug_spe, aug_gen, keys_tensor, shape, cond_scale = 6., rescaled_phi = 0.7):
        batch, device = shape[0], self.betas.device
         ######## 1.2
        img = torch.randn(shape, device=device) * 1.2

        x_start = None

        for t in tqdm(reversed(range(0, self.num_timesteps)), desc = 'sampling loop time step', total = self.num_timesteps):
            img, x_start = self.p_sample(img, t, classes,  combination_dict, classes_gen, combination_dict_gen,  aug_spe, aug_gen,keys_tensor, cond_scale, rescaled_phi)
            assert img.shape[0]==batch or img.shape[0]==batch//2


        # img = unnormalize_to_zero_to_one(img)
        return img

    @torch.no_grad()
    def ddim_sample(self, classes, combination_dict, classes_gen, combination_dict_gen, aug_spe, aug_gen, labels, shape, cond_scale = 6, rescaled_phi = 0.7, clip_denoised = True):
        batch, device, total_timesteps, sampling_timesteps, eta, objective = shape[0], self.betas.device, self.num_timesteps, self.sampling_timesteps, self.ddim_sampling_eta, self.objective

        times = torch.linspace(-1, total_timesteps - 1, steps=sampling_timesteps + 1)   # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:])) # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]

        img = torch.randn(shape, device = device)

        x_start = None

        for time, time_next in tqdm(time_pairs, desc = 'sampling loop time step'):
            time_cond = torch.full((batch,), time, device=device, dtype=torch.long)
            pred_noise, x_start, *_ = self.model_predictions(img, time_cond, classes, combination_dict, classes_gen, combination_dict_gen,aug_spe, aug_gen, labels, cond_scale = cond_scale, rescaled_phi = rescaled_phi, clip_x_start = clip_denoised)

            if time_next < 0:
                img = x_start
                continue

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]

            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()

            noise = torch.randn_like(img)

            img = x_start * alpha_next.sqrt() + \
                  c * pred_noise + \
                  sigma * noise
            if torch.isnan(img).any().item():
                print("nan")
        # img = unnormalize_to_zero_to_one(img)
        return img



    # def ddim_sample(self, classes, shape, cond_scale = 6., rescaled_phi = 0.7, clip_denoised = True):
    #     batch, device, total_timesteps, sampling_timesteps, eta, objective = shape[0], self.betas.device, self.num_timesteps, self.sampling_timesteps, self.ddim_sampling_eta, self.objective

    #     times = torch.linspace(-1, total_timesteps - 1, steps=sampling_timesteps + 1)   # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
    #     times = list(reversed(times.int().tolist()))
    #     time_pairs = list(zip(times[:-1], times[1:])) # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]

    #     img = torch.randn(shape, device = device)

    #     x_start = None

    #     for time, time_next in tqdm(time_pairs, desc = 'sampling loop time step'):
    #         time_cond = torch.full((batch,), time, device=device, dtype=torch.long)
    #         pred_noise, x_start, *_ = self.model_predictions(img, time_cond, classes, cond_scale = cond_scale, rescaled_phi = rescaled_phi, clip_x_start = clip_denoised)

    #         if time_next < 0:
    #             img = x_start
    #             continue

    #         alpha = self.alphas_cumprod[time]
    #         alpha_next = self.alphas_cumprod[time_next]

    #         sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
    #         c = (1 - alpha_next - sigma ** 2).sqrt()

    #         noise = torch.randn_like(img)

    #         img = x_start * alpha_next.sqrt() + \
    #               c * pred_noise + \
    #               sigma * noise

    #     # img = unnormalize_to_zero_to_one(img)
    #     return img

    @torch.no_grad()
    def sample(self, classes, combination_dict, classes_gen, combination_dict_gen, aug_spe, aug_gen, keys_tensor, cond_scale = 6., rescaled_phi = 0.7):
        if len(classes) == 0:
            batch_size, seq_length, channels = len(aug_spe), self.seq_length, self.channels
            if batch_size == 0:
                batch_size, seq_length, channels = len(aug_gen), self.seq_length, self.channels
                if batch_size == 0:
                    batch_size, seq_length, channels = len(classes_gen), self.seq_length, self.channels
       
        else:
            batch_size, seq_length, channels = classes.shape[0], self.seq_length, self.channels
        batch_size = len(combination_dict)
        sample_fn = self.p_sample_loop if not self.is_ddim_sampling else self.ddim_sample
        return sample_fn(classes, combination_dict, classes_gen, combination_dict_gen, aug_spe, aug_gen, keys_tensor, (batch_size, channels, seq_length), cond_scale, rescaled_phi)

    @torch.no_grad()
    def interpolate(self, x1, x2, classes, combination_dict, classes_gen, combination_dict_gen, aug_spe, aug_gen, keys_tensor,cond_scale = 6., rescaled_phi = 0.7, t = None, lam = 0.5):

        b, *_, device = *x1.shape, x1.device
        t = default(t, self.num_timesteps - 1)

        assert x1.shape == x2.shape

        t_batched = torch.stack([torch.tensor(t, device = device)] * b)
        xt1, xt2 = map(lambda x: self.q_sample(x, t = t_batched), (x1, x2))


        img = (1 - lam) * xt1 + lam * xt2

        for i in tqdm(reversed(range(0, t)), desc = 'interpolation sample time step', total = t):
            img, _ = self.p_sample(img, t, classes,  combination_dict, classes_gen, combination_dict_gen,  aug_spe, aug_gen,keys_tensor, cond_scale, rescaled_phi)
        return img



    def split_and_scale_fixed_range(self, x, repeat, fixed_scale_factor=1.2):
        # Check if the batch size is divisible by repeat
        assert x.shape[0] % repeat == 0, "Batch size should be divisible by repeat"

        # Calculate the size of each split
        split_size = x.shape[0] // repeat

        # Clip the fixed scaling factor to be within [1.0, 1.5]
        fixed_scale_factor = max(1.0, min(fixed_scale_factor, 1.5))

        # Repeat the fixed scaling factor to match the batch size
        scaling_factors = torch.full((repeat,), fixed_scale_factor, device=x.device)
        scaling_factors = scaling_factors.repeat_interleave(split_size)


        assert x.shape[0] % repeat == 0, "Length L should be divisible by repeat"
        min_value = 1.0
        max_value = 1.5
        L  = x.shape[0]

        scaling_factors = torch.ones(x.shape[0])

        # Calculate the interval between each portion
        interval = (max_value - min_value) / (repeat - 1)

        
        for i in range(1, repeat):
            startlen = (x.shape[0] // repeat) * i
            endlen = (x.shape[0] // repeat) * (i+1)
            scaling_factors[startlen: endlen] = scaling_factors[startlen-1]+interval



        self.vary_noise = scaling_factors .unsqueeze(1).unsqueeze(2).cuda()

    @autocast(enabled = False)
    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start) ) 
        # if self.repeat:
        #     self.split_and_scale_fixed_range(noise, self.repeat)
        #     noise = noise * self.vary_noise

        if self.offset_noise_strength > 0.:
            offset_noise = torch.randn(x_start.shape[:2], device = self.device)
            noise += self.offset_noise_strength * rearrange(offset_noise, 'b c -> b c 1 1')

        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def p_losses(self, x_start, t, *, classes, noise = None):
        b, c, h= x_start.shape
        noise = default(noise, lambda: torch.randn_like(x_start))

        # noise sample

        x = self.q_sample(x_start = x_start, t = t, noise = noise)

        # predict and take gradient step

        model_out = self.model(x, t, classes)

        if self.objective == 'pred_noise':
            target = noise
        elif self.objective == 'pred_x0':
            target = x_start
        elif self.objective == 'pred_v':
            v = self.predict_v(x_start, t, noise)
            target = v
        else:
            raise ValueError(f'unknown objective {self.objective}')

        loss = F.mse_loss(model_out, target, reduction = 'none')
        loss = reduce(loss, 'b ... -> b', 'mean')

        loss = loss * extract(self.loss_weight, t, loss.shape)
        return loss.mean()

    def forward(self, img, *args, **kwargs):
        b, c, h, device, seq_length, = *img.shape, img.device, self.seq_length
        assert h == seq_length , f'height of time series must be {seq_length}'
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()

        # img = normalize_to_neg_one_to_one(img)
        return self.p_losses(img, t, *args, **kwargs)
