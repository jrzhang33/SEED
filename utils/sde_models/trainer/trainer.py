import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch
import numpy as np

class NTXentLoss(torch.nn.Module):

    def __init__(self, device, batch_size, temperature, use_cosine_similarity):
        super(NTXentLoss, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.device = device
        self.softmax = torch.nn.Softmax(dim=-1)
        self.mask_samples_from_same_repr = self._get_correlated_mask().type(torch.bool)
        self.similarity_function = self._get_similarity_function(use_cosine_similarity)
        self.criterion = torch.nn.CrossEntropyLoss(reduction="sum")

    def _get_similarity_function(self, use_cosine_similarity):
        if use_cosine_similarity:
            self._cosine_similarity = torch.nn.CosineSimilarity(dim=-1)
            return self._cosine_simililarity
        else:
            return self._dot_simililarity

    def _get_correlated_mask(self):
        diag = np.eye(2 * self.batch_size)
        l1 = np.eye((2 * self.batch_size), 2 * self.batch_size, k=-self.batch_size)
        l2 = np.eye((2 * self.batch_size), 2 * self.batch_size, k=self.batch_size)
        mask = torch.from_numpy((diag + l1 + l2))
        mask = (1 - mask).type(torch.bool)
        return mask.to(self.device)

    @staticmethod
    def _dot_simililarity(x, y):
        v = torch.tensordot(x.unsqueeze(1), y.T.unsqueeze(0), dims=2)
        # x shape: (N, 1, C)
        # y shape: (1, C, 2N)
        # v shape: (N, 2N)
        return v

    def _cosine_simililarity(self, x, y):
        # x shape: (N, 1, C)
        # y shape: (1, 2N, C)
        # v shape: (N, 2N)
        v = self._cosine_similarity(x.unsqueeze(1), y.unsqueeze(0))
        return v

    def forward(self, zis, zjs):
        representations = torch.cat([zjs, zis], dim=0)
        self.batch_size = zjs.shape[0]
        self.mask_samples_from_same_repr = self._get_correlated_mask().type(torch.bool)

        similarity_matrix = self.similarity_function(representations, representations) 
        
        # filter out the scores from the positive samples 
        l_pos = torch.diag(similarity_matrix, self.batch_size) 
        r_pos = torch.diag(similarity_matrix, -self.batch_size)
        positives = torch.cat([l_pos, r_pos]).view(2 * self.batch_size, 1) 
        negatives = similarity_matrix[self.mask_samples_from_same_repr].view(2 * self.batch_size, -1) 
        logits = torch.cat((positives, negatives), dim=1) 
        logits /= self.temperature

        labels = torch.zeros(2 * self.batch_size).to(self.device).long() 
        loss = self.criterion(logits, labels)

        return loss / (2 * self.batch_size)
    
def Trainer(model, temporal_contr_model, model_optimizer, temp_cont_optimizer, train_dl, valid_dl, test_dl, device, logger, config, experiment_log_dir, training_mode, testuser):
    logger.debug("Training started ....")
    criterion = nn.CrossEntropyLoss()
    for epoch in range(config.num_epoch):
        last_epoch = (epoch == config.num_epoch - 1)
        train_loss, train_acc, _ = model_train(
            model, temporal_contr_model, model_optimizer, temp_cont_optimizer,
            criterion, train_dl, config, device, training_mode, last_epoch, epoch, testuser['name']
        )


        logger.debug(f'\nEpoch : {epoch}\nTrain Loss     : {train_loss:.4f}\t | \tTrain Accuracy     : {train_acc:2.4f}\n')
        chkpoint = {
            'model_state_dict': model.state_dict(),
            'temporal_contr_model_state_dict': temporal_contr_model.state_dict()
        }
        torch.save(chkpoint, os.path.join(
            os.getcwd(), 'intermediate_results', f"{testuser['name']}-sde.pt")
        )

    logger.debug("\n################## Training is Done! #########################")

def model_train(model, temporal_contr_model, model_optimizer, temp_cont_optimizer, criterion, train_loader, config, device, training_mode, last_epoch, epoch, name):
    L_dis_fn = NTXentLoss(device, config.batch_size, config.Context_Cont.temperature, config.Context_Cont.use_cosine_similarity)
    total_loss, total_acc = [], []
    model.train()
    temporal_contr_model.train()

    if last_epoch:
        model.eval()
        temporal_contr_model.eval()

    for minibatch in train_loader:
        x = minibatch[2].cuda().float()
        y = minibatch[1].cuda().long()
        if x.dim() == 3:
            x = x.unsqueeze(3)
        data = x.squeeze(3).float().to(device)
        y = y.to(device)

        model_optimizer.zero_grad()
        temp_cont_optimizer.zero_grad()

        features1, features2 = model(data, 'binomial')
        features1 = F.normalize(features1, dim=1)
        features2 = F.normalize(features2, dim=1)

        _, s_feat1, c_feat1, dom_pred1, aux_pred1, s_t1, c_t1 = temporal_contr_model(features1, features2)
        _, s_feat2, c_feat2, dom_pred2, aux_pred2, s_t2, c_t2 = temporal_contr_model(features2, features1)

        L_dom = F.cross_entropy(dom_pred1, y) + F.cross_entropy(dom_pred2, y)
        L_intra = cosine_pairwise_loss(s_feat1) + cosine_pairwise_loss(s_feat2)
        L_cls = info_nce_loss(c_feat1, y) + info_nce_loss(c_feat2, y)
        L_ort = frobenius_norm(s_t1, c_t1) + frobenius_norm(s_t2, c_t2)
        L_aux = F.cross_entropy(aux_pred1, y) + F.cross_entropy(aux_pred2, y)
        L_dis = L_dis_fn(s_feat1, s_feat2)

        loss = (
            0.01 * L_dis +
            0.1 * L_dom +
            L_cls + L_aux +
            1.2 * L_ort +       
            1.4 * L_intra      
        )

        total_loss.append(loss.item())
        loss.backward()
        model_optimizer.step()
        temp_cont_optimizer.step()

    total_loss = torch.tensor(total_loss).mean()
    total_acc = 0
    return total_loss, total_acc, []


def cosine_pairwise_loss(tensor):
    normalized = F.normalize(tensor, p=2, dim=1)
    cosine_sim = torch.matmul(normalized, normalized.T)
    batch_size = cosine_sim.size(0)
    mask = torch.eye(batch_size, device=tensor.device).bool()
    cosine_sim = cosine_sim.masked_fill(mask, 0)
    return torch.mean(torch.abs(cosine_sim))

def frobenius_norm(a, b):
    a = F.normalize(a, p=2, dim=1)
    b = F.normalize(b, p=2, dim=1)
    sim_matrix = torch.matmul(a, b.T)
    return torch.norm(sim_matrix, p='fro') ** 2

def info_nce_loss(features, labels, temperature=0.1):
    features = F.normalize(features, p=2, dim=1)
    similarity_matrix = torch.matmul(features, features.T)
    logits = similarity_matrix / temperature
    batch_size = features.size(0)
    labels = labels.unsqueeze(1)
    same_class_mask = (labels == labels.T).float()
    self_mask = torch.eye(batch_size, device=features.device)
    same_class_mask = same_class_mask - self_mask
    exp_logits = torch.exp(logits)
    pos_logits = exp_logits * same_class_mask
    denominator = torch.sum(exp_logits, dim=1, keepdim=True)
    numerator = torch.sum(pos_logits, dim=1, keepdim=True)
    loss = -torch.log(numerator / (denominator + 1e-8) + 1e-8)
    return torch.mean(loss)
