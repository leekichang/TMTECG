"""
Created on Thu Aug 20 2023
@author: Kichang Lee
@contact: kichang.lee@yonsei.ac.kr
"""
__all__ = [
    'OURS',
    ]

import os
import copy
import queue
import torch
import random
import threading
import numpy as np
from tqdm import tqdm
import torch.nn as nn
from datetime import datetime
from torch.utils.data import DataLoader

import ops
import utils

import torch.utils.tensorboard as tb
class OURSLoss(nn.Module):
    def __init__(self, temperature=0.5):
        super(OURSLoss, self).__init__()
        self.temperature = temperature
    
    def forward(self, z_i, z_j, time, patient_id):
        B   = patient_id.shape
        z_i = nn.functional.normalize(z_i, dim=1)
        z_j = nn.functional.normalize(z_j, dim=1)
        
        # Cosine similarity between normalized embeddings
        similarity_matrix = torch.matmul(z_i, z_j.T) / self.temperature

        id_mask        = (patient_id[:, None] == patient_id[None, :])
        time_threshold = 12 # Take the samples in +-120sec as positive pairs
        time_mask      = (torch.abs(time[:, None] - time[None, :]) <= time_threshold)   

        final_mask = id_mask & time_mask
        count_true = final_mask.sum().item()

        print(f'count_true:{count_true} z_i.shape:{z_i.shape}')
        
        # Calculate positive and negative pairs
        batch_size    = z_i.size(0)
        positive_mask = torch.eye(batch_size, dtype=torch.bool, device=z_i.device)
        negative_mask = ~positive_mask
        
        # Compute loss
        positive_pairs = similarity_matrix[positive_mask].view(batch_size, -1)
        negative_pairs = similarity_matrix[negative_mask].view(batch_size, -1)
        
        # Calculate logits and labels -> self-contrastive loss
        logits = torch.cat([positive_pairs, negative_pairs], dim=1)
        labels = torch.zeros(batch_size, device=z_i.device, dtype=torch.long)
        
        # Calculate cross-entropy loss
        loss1 = nn.functional.cross_entropy(logits, labels)
        zero_fill = similarity_matrix
        zero_fill[final_mask==False] = 0
        # loss w.r.t. the temporal invariance
        loss2 = -1*torch.mean(torch.log(torch.sum(torch.exp(zero_fill), dim=1)/torch.sum(torch.exp(similarity_matrix), dim=1)))
        
        loss  = loss1+loss2
        
        return loss
        
    
class OURS:
    def __init__(self, args):
        self.args         = args
        self.save_path    = f'./checkpoints/{args.exp_name}'
        os.makedirs(self.save_path, exist_ok=True)
        
        self.epoch        = 0
        self.epochs       = args.epochs
        self.device       = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.model        = utils.build_model(args).to(self.device)
        self.model.classifier = nn.Identity().to(self.device)
        
        self.criterion    = OURSLoss(args.t).to(self.device)
        self.optimizer    = utils.build_optimizer(self.model, args)
        
        self.dataset      = utils.load_dataset(args) #None # assigned in self.train()
        self.dataset.setup()
        
        self.data_shape   = (-1, 12, 2500)
        
        self.dataloader   = None # assigned in self.train()
        self.augmentator  = ops.Augmentator()
        
        self.train_loss = []
        
        if self.args.use_tb:
            self.TB_WRITER = tb.SummaryWriter(f'./tensorboard/{str(datetime.now().strftime("%Y-%m-%d_%H:%M:%S"))}_{self.args.exp_name}')
        
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f'model name:{args.model}\ndataset:{args.dataset}\ndevice:{self.device}\nTotal parameter:{total_params:,}')

    def train(self):
        self.model.train()
        losses = []
        
        for chunk_idx in tqdm(range(self.dataset.num_chunks)):
            data_queue = queue.Queue()
            completion_event = threading.Event()
            background_thread = threading.Thread(target=utils.background_loading, \
                args=(self.dataset.chunk_idx, self.dataset.num_chunks, self.dataset.subjects[chunk_idx], data_queue, self.data_shape))
            background_thread.daemon = True
            background_thread.start()
            self.dataloader = DataLoader(self.dataset, batch_size=self.args.batch_size, shuffle=True, drop_last=True)
            for (X, id_, length) in self.dataloader:
                self.optimizer.zero_grad()
                X      = X.to(self.device)
                id_    = id_.to(self.device)
                length = length.to(self.device)
                X1, augs = self.augmentator(X, None)
                X2, _    = self.augmentator(X, augs) # -> Need to Augment in (B,C,S) -> (2*B,C,S)
                del X
                
                z1 = self.model.projector(self.model(X1))
                z2 = self.model.projector(self.model(X2)) # calculate each feature
                
                loss = self.criterion(z1, z2, length, id_)
                loss.backward()
                self.optimizer.step()
                losses.append(loss.item())
            chunk_idx, next_data = data_queue.get()
            self.dataset.chunk_idx = chunk_idx
            self.dataset.next_data = next_data
            self.dataset.update()
                  
        self.train_loss.append(np.mean(losses))
        if self.args.use_tb:
            self.TB_WRITER.add_scalar("Train Loss", np.mean(losses), self.epoch+1)

    @torch.no_grad()
    def test(self):
        return
    
    def save_model(self):
        torch.save(self.model.state_dict(), f'{self.save_path}/{self.epoch+1}.pth')

    def print_train_info(self):
        print(f'({self.epoch+1:03}/{self.epochs}) Train Loss:{self.train_loss[self.epoch]:>6.4f}', flush=True)

if __name__ == '__main__':
    from tqdm import tqdm
    args = utils.parse_args()
    torch.manual_seed(args.seed)
    trainer = OURS(args)
    for epoch in tqdm(range(trainer.epochs)):
        trainer.train()
        trainer.test()
        trainer.print_train_info()
        trainer.save_model()
        trainer.epoch += 1