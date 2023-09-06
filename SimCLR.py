"""
Created on Thu Aug 20 2023
@author: Kichang Lee
@contact: kichan.lee@yonsei.ac.kr
"""
__all__ = [
    'SimCLR',
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
from sklearn.metrics import recall_score, f1_score
import ops
import utils

import torch.utils.tensorboard as tb
class SimCLRLoss(nn.Module):
    def __init__(self, temperature=0.5):
        super(SimCLRLoss, self).__init__()
        self.temperature = temperature
    
    def forward(self, z_i, z_j):
        z_i = nn.functional.normalize(z_i, dim=1)
        z_j = nn.functional.normalize(z_j, dim=1)
        
        # Cosine similarity between normalized embeddings
        similarity_matrix = torch.matmul(z_i, z_j.T) / self.temperature
        
        # Calculate positive and negative pairs
        batch_size = z_i.size(0)
        positive_mask = torch.eye(batch_size, dtype=torch.bool, device=z_i.device)
        negative_mask = ~positive_mask
        
        # Compute loss
        positive_pairs = similarity_matrix[positive_mask].view(batch_size, -1)
        negative_pairs = similarity_matrix[negative_mask].view(batch_size, -1)
        
        # Calculate logits and labels
        logits = torch.cat([positive_pairs, negative_pairs], dim=1)
        labels = torch.zeros(batch_size, device=z_i.device, dtype=torch.long)
        
        # Calculate cross-entropy loss
        loss = nn.functional.cross_entropy(logits, labels)
        
        return loss
        
    
class SimCLR:
    def __init__(self, args):
        self.args         = args
        self.save_path    = f'./checkpoints/{args.exp_name}'
        os.makedirs(self.save_path, exist_ok=True)
        
        self.epoch        = 0
        self.epochs       = args.epochs
        self.device       = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.model        = utils.build_model(args).to(self.device)
        
        self.criterion    = SimCLRLoss(args.t).to(self.device)
        self.optimizer    = utils.build_optimizer(self.model, args)
        
        self.dataset      = utils.load_dataset(args) #None # assigned in self.train()
        self.dataset.setup()
        
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
                args=(self.dataset.chunk_idx, self.dataset.num_chunks, self.dataset.subjects[chunk_idx], data_queue))
            background_thread.daemon = True
            background_thread.start()
            self.dataloader = DataLoader(self.dataset, batch_size=self.args.batch_size, shuffle=True, drop_last=False)
            for X in self.dataloader:
                self.optimizer.zero_grad()
                X = X.to(self.device)
                X1, augs = self.augmentator(X, None)
                X2, _    = self.augmentator(X, augs) # -> Need to Augment in (B,C,S) -> (2*B,C,S)
                del X
                
                z1 = self.model.projector(self.model(X1))
                z2 = self.model.projector(self.model(X2)) # calculate each feature
                
                loss = self.criterion(z1, z2)                   
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
    trainer = SimCLR(args)
    for epoch in tqdm(range(trainer.epochs)):
        trainer.train()
        trainer.test()
        trainer.print_train_info()
        trainer.save_model()
        trainer.epoch += 1