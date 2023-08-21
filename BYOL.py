"""
Created on Thu Aug 18 2023
@author: Kichang Lee
@contact: kichan.lee@yonsei.ac.kr
"""
__all__ = [
    'BYOL',
    ]

import os
import copy
import torch
import random
import numpy as np
import torch.nn as nn
from functools import wraps
from datetime import datetime
from torch.utils.data import DataLoader
from sklearn.metrics import recall_score, f1_score
from tqdm import tqdm
import ops
import utils

import torch.utils.tensorboard as tb

class BYOLLoss(nn.Module):
    def __init__(self, temperature=0.5):
        super(BYOLLoss, self).__init__()
        self.temperature = temperature

    def forward(self, z_i, z_j):
        z_i_norm = nn.functional.normalize(z_i, dim=1, p=2)
        z_j_norm = nn.functional.normalize(z_j, dim=1, p=2)
        loss = 2 - 2 * (z_i_norm * z_j_norm).sum(dim=-1)
        return loss.mean()

# exponential moving average

class EMA():
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

class BYOL:
    def __init__(self, args):
        self.args         = args
        self.save_path    = f'./checkpoints/{args.exp_name}'
        os.makedirs(self.save_path, exist_ok=True)
        
        self.epoch        = 0
        self.epochs       = args.epochs
        self.device       = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.online_model = utils.build_model(args).to(self.device)
        self.target_model = utils.build_model(args).to(self.device)
        utils.set_requires_grad(self.target_model, False)
        
        self.ema_updater = EMA(args.ma_decay)
        
        
        self.criterion    = BYOLLoss(args.t).to(self.device)
        self.optimizer    = utils.build_optimizer(self.online_model, args)
        
        self.dataset      = None # assigned in self.train()
        self.dataloader   = None # assigned in self.train()
        self.augmentator  = ops.Augmentator()
        
        self.data_chunks  = [file for file in os.listdir(f'./dataset/TMT_unlabeled') if file.endswith('.npz')]

        self.train_loss = 0
        
        self.TB_WRITER = tb.SummaryWriter(f'./tensorboard/{str(datetime.now().strftime("%Y-%m-%d_%H:%M:%S"))}_{self.args.exp_name}')
        
        total_params = sum(p.numel() for p in self.online_model.parameters())
        print(f'model name:{args.model}\ndataset:{args.dataset}\ndevice:{self.device}\nTotal parameter:{total_params:,}')

    def target_update(self):
        for current_params, ma_params in zip(self.online_model.parameters(), self.target_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.ema_updater.update_average(old_weight, up_weight)

    def train(self):
        self.target_model.eval()
        self.online_model.train()
        losses = []
        chunk_idxs = [idx+1 for idx in range(len(self.data_chunks))]
        random.shuffle(chunk_idxs)          # shuffle chunk index to mitigate sequential bias
        for chunk_idx in tqdm(chunk_idxs):
            self.args.stage = chunk_idx
            self.dataset    = utils.load_dataset(args, is_train=True)
            self.dataloader = DataLoader(self.dataset, batch_size=self.args.batch_size, shuffle=True, drop_last=True)
            for X in self.dataloader:
                self.optimizer.zero_grad()
                X = X.to(self.device)
                X1, augs = self.augmentator(X, None) 
                X2, _    = self.augmentator(X, augs)
                del X
                del augs
                
                z_i_1 = self.online_model.predictor(self.online_model(X1))
                z_i_2 = self.online_model.predictor(self.online_model(X2))
                
                with torch.no_grad():
                    z_j_1 = self.target_model(X2)
                    z_j_2 = self.target_model(X1)
                    
                loss1 = self.criterion(z_i_1, z_j_1)
                loss2 = self.criterion(z_i_2, z_j_2)
                loss = loss1+loss2
                
                loss.backward()
                self.optimizer.step()
                self.target_update()
                losses.append(loss.item())
        self.train_loss = np.mean(losses)
        self.TB_WRITER.add_scalar("Train Loss", self.train_loss, self.epoch+1)
    
    @torch.no_grad()
    def test(self):
        return
    
    def save_model(self):
        torch.save(self.online_model.state_dict(), f'{self.save_path}/{self.epoch+1}.pth')

    def print_train_info(self):
        print(f'({self.epoch+1:03}/{self.epochs}) Train Loss:{self.train_loss:>6.4f}', flush=True)

if __name__ == '__main__':
    from tqdm import tqdm
    args = utils.parse_args()
    torch.manual_seed(args.seed)
    trainer = BYOL(args)
    for epoch in tqdm(range(trainer.epochs)):
        trainer.train()
        trainer.test()
        trainer.print_train_info()
        trainer.save_model()
        trainer.epoch += 1