import os
import torch
import math
import argparse
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import torch.nn as nn
from datautil.getdataloader_single import get_act_dataloader #Follow the official Library: robustlearn
from adarnn.base.loss_transfer import TransferLoss #Follow the official Library: adarnn
from utils.sae.train_sae_seg import Trainer1D as Trainer1D 
from utils.sae.codes.models.scaling_autoencoder import * # Follow: GENERATIVE LEARNING FOR FINANCIAL TIME SERIES WITH IRREGULAR AND SCALE-INVARIANT PATTERNS
from utils.sae.codes.models.pattern_generation_module import * #~
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def TDC(K_segment, dataloader,_, dis_type='coral'): 

    try:
        dataset = dataloader.dataset.x_data 
        data = dataset.cpu().numpy().transpose(0, 2, 1) 
    except:
        dataset = dataloader.dataset.x 
        if len(dataset.shape) == 4:
            dataset = dataset.squeeze(2)
            try:
                data = dataset.numpy().transpose(0, 2, 1)
            except:
                data = dataset.transpose(0, 2, 1)

    start_time = 0
    end_time = dataset.shape[-1]
    
    
    num_day = end_time - start_time
   
    feat = data # #batch, len, dim
    feat = torch.tensor(feat, dtype=torch.float32)
    feat_shape_1 = feat.shape[1] 
    feat = feat.reshape(-1, feat.shape[2])
    feat = feat.cuda()
    split_N  =10

    selected = [0, 10]
    candidate = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    start = 0

    if K_segment in [2, 3, 5, 7, 10]: 
         
        while len(selected) - 2 < K_segment - 1:
            distance_list = []
            for can in candidate:
                selected.append(can)
                selected.sort()
                dis_temp = 0
                for i in range(1, len(selected) - 1):
                    for j in range(i, len(selected) - 1):
                        index_part1_start = start + math.floor(selected[i-1] / split_N * num_day) * feat_shape_1
                        index_part1_end = start + math.floor(selected[i] / split_N * num_day) * feat_shape_1
                        feat_part1 = feat[index_part1_start: index_part1_end]
                        index_part2_start = start + math.floor(selected[j] / split_N * num_day) * feat_shape_1
                        index_part2_end = start + math.floor(selected[j+1] / split_N * num_day) * feat_shape_1
                        feat_part2 = feat[index_part2_start: index_part2_end]
                        criterion_transder = TransferLoss(loss_type=dis_type, input_dim=feat_part1.shape[1])
                        dis_temp += criterion_transder.compute(feat_part1, feat_part2)
                distance_list.append(dis_temp)
                selected.remove(can)
            can_index = distance_list.index(max(distance_list))
            selected.append(candidate[can_index])
            candidate.remove(candidate[can_index]) 
   
        selected.sort()
        res = []  
        for i in range(1, len(selected)):
            sel_start_time = int(num_day / split_N * selected[i - 1])
            sel_end_time = int(num_day / split_N * selected[i])
            res.append((sel_start_time, sel_end_time))
        return res
    else:
        print("error in number of domain")

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
        return  self.ori_x[idx], self.ori_y[idx], self.packed_data[idx], self.lengths[idx],self.sample_idx, idx
def eval_generation_module(name,
    pgm,dataset,
    dataloader,
    device=None,
):
    
    pgm.to(device)

    pgm.eval()
    with torch.no_grad():
        for data in dataloader:
            x, lengths, orix, oriy,_,indices = data
            x = x.to(dtype=torch.float32)
            x = x.to(device)
            batch_size = x.shape[0]

            x_0_,  z = pgm.forward(x,lengths)
            #replace x with z
            z = z.transpose(2,1).cpu()

            dataset.packed_data = dataset.packed_data.to(dtype=torch.float32)
            dataset.packed_data[indices] = z

    
    dataset_dict = {
        'packed_data': dataset.packed_data,
        'lengths': dataset.lengths,
        'ori_x': dataset.ori_x,
        'ori_y': dataset.ori_y,
        'sample_idx': dataset.sample_idx
    }
    save_dir = 'intermediate_results'
    os.makedirs(save_dir, exist_ok=True)
    filename = f"{name}-segment.pth"
    save_path = os.path.join(save_dir, filename)

    torch.save(dataset_dict, save_path)
   



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='emg', type=str)
    parser.add_argument('--task_id', default=1, type=int)
    parser.add_argument('--run_id', default=0, type=int)
    parser.add_argument('--segment_K', default=5, type=int)
    args = parser.parse_args()
    testuser = {}
    testuser['dataset'] = args.dataset
    testuser['name'] = f"{args.dataset}_task_{args.task_id}_seed_{args.run_id}"

    _, _, _,  train_loader= get_act_dataloader(args)
    source_loaders = train_loader

    segments = TDC(args.segment_K, source_loaders, 'Changping', 'coral')
    packed_data = []
    lengths = []
    ori_x = []
    ori_y = []
    sample_idx = []
    max_length = 0
    try:
        x_data = source_loaders.dataset.x_data
        y_data = source_loaders.dataset.y_data
    except:
        x_data = torch.tensor(source_loaders.dataset.x.squeeze(2)) 
        try:
            y_data = torch.tensor(source_loaders.dataset.labels)
        except:
            y_data = torch.tensor(source_loaders.dataset.label)
    for i,data in enumerate(x_data): #data:dim, len
        
        for ss in segments:
            start, end = ss
            segment = data[:, start:end]
            packed_data.append(segment)
            ll = end - start
            lengths.append(end - start)
            ori_x.append(data)
            ori_y.append(y_data[i])
            sample_idx.append(i)
            if ll > max_length:
                max_length = ll
    for i in range(len(packed_data)):
        # padded_segment = packed_data[i].unsqueeze(0)
        padded_segment = torch.nn.functional.pad(packed_data[i], (0, max_length - lengths[i])).unsqueeze(0)
        if i == 0:
            packed_array = padded_segment
        else:
            packed_array = torch.cat((packed_array, padded_segment), 0) 
   
    dataset = CustomDataset(packed_array, lengths, ori_x, ori_y,sample_idx)
    source_loaders = DataLoader(dataset, batch_size=64, shuffle=False)
    for minibatch in source_loaders: 
        batch_size = minibatch[0].shape[0]
        model_dim = minibatch[2].shape[-1] * minibatch[2].shape[-2]
        print("print shape X:",minibatch[0].shape)
        shapex =[minibatch[0].shape[0],minibatch[0].shape[2],minibatch[0].shape[1]]#length,channel
        break
    sae= ScalingAE(shapex[2],shapex[2],shapex[2] )
    fc = nn.Linear(model_dim ,6)
    pgm = PatternGenerationModule(sae, fc, condition=True, device=device)
    trainer = Trainer1D(
        pgm,
        None,
        dataloader = source_loaders,
        train_batch_size = shapex[0],
        train_lr =  2e-3, 
        train_num_steps = 800,      
        gradient_accumulate_every = 2,   
        ema_decay = 0.995,                
        amp = False,                      
        
    )

    trainer.train(testuser)
    eval_generation_module(testuser['name'],
        pgm,dataset,
        source_loaders,
        device=device)

if __name__ == "__main__":
    main()
