"""
Created on Thu Aug 17 2023
@author: Kichang Lee
@contact: kichan.lee@yonsei.ac.kr
"""
__all__ = [
    'SupervisedTrainer',
    ]

import os
import torch
import numpy as np
from datetime import datetime
from torch.utils.data import DataLoader
from sklearn.metrics import recall_score, f1_score

import utils

import torch.utils.tensorboard as tb

class SupervisedTrainer:
    def __init__(self, args):
        self.args         = args
        self.save_path    = f'./checkpoints/{args.exp_name}'
        os.makedirs(self.save_path, exist_ok=True)
        
        self.epoch        = 0
        self.epochs       = args.epochs
        self.device       = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.model        = utils.build_model(args).to(self.device)
        self.criterion    = utils.build_criterion(args).to(self.device)
        self.optimizer    = utils.build_optimizer(self.model, args)
        
        self.trainset     = utils.load_dataset(args, is_train=True)
        self.testset      = utils.load_dataset(args, is_train=False)
        
        self.train_loader = DataLoader(self.trainset, batch_size=args.batch_size, shuffle=True , drop_last=True )
        self.test_loader  = DataLoader(self.testset , batch_size=args.batch_size, shuffle=False, drop_last=False)
    
        self.train_loss = []
        self.test_loss  = []
        self.accs       = []
        self.recalls    = []
        self.f1s        = []
        self.specs      = []
        
        self.TB_WRITER = tb.SummaryWriter(f'./tensorboard/{str(datetime.now().strftime("%Y-%m-%d_%H:%M:%S"))}_{self.args.exp_name}')
        
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f'model name:{args.model}\ndataset:{args.dataset}\ndevice:{self.device}\nTotal parameter:{total_params:,}')

    def train(self):
        self.model.train()
        losses = []
        for X, Y in self.train_loader:
            self.optimizer.zero_grad()
            X, Y = X.to(self.device), Y.to(self.device)
            pred = self.model(X)
            loss = self.criterion(pred, Y)
            losses.append(loss.item())
            loss.backward()
            self.optimizer.step()
        self.train_loss.append(np.mean(losses))
        self.TB_WRITER.add_scalar("Train Loss", np.mean(losses), self.epoch+1)
        
    @torch.no_grad()
    def test(self):
        self.model.eval()
        preds, targets, losses = [], [], []
        for X, Y in self.test_loader:
            X, Y = X.to(self.device), Y.to(self.device)
            pred = self.model(X)
            loss = self.criterion(pred, Y)
            
            preds.append(pred.cpu().numpy())
            targets.append(Y.cpu().numpy())
            losses.append(loss.item())
            
        preds = np.concatenate(preds)
        targets = np.concatenate(targets)
        
        acc    = utils.calculate_topk_accuracy(torch.from_numpy(preds), torch.from_numpy(targets), k=1)
        recall = recall_score(y_true=targets, y_pred=np.argmax(preds, axis=-1))
        f1     = f1_score(y_true=targets, y_pred=np.argmax(preds, axis=-1))
        spec   = utils.specificity_score(y_true=targets, y_pred=np.argmax(preds, axis=-1))

        
        self.test_loss.append(np.mean(losses))
        self.accs.append(acc)
        self.recalls.append(recall)
        self.f1s.append(f1)
        self.specs.append(spec)
        self.TB_WRITER.add_scalar(f'Test Loss', np.mean(losses), self.epoch+1)
        self.TB_WRITER.add_scalar(f'Test Accuracy', acc, self.epoch+1)
        self.TB_WRITER.add_scalar(f'Recall', recall, self.epoch+1)
        self.TB_WRITER.add_scalar(f'F1-Score', f1, self.epoch+1)
        self.TB_WRITER.add_scalar(f'Specificity', spec, self.epoch+1)
    
    def save_model(self):
        torch.save(self.model.state_dict(), f'{self.save_path}/{self.epoch+1}.pth')

    def print_train_info(self):
        print(f'({self.epoch+1:03}/{self.epochs}) Train Loss:{self.train_loss[self.epoch]:>6.4f} Test Loss:{self.test_loss[self.epoch]:>6.4f} Test Accuracy:{self.accs[self.epoch]:>5.2f}%', flush=True)
        print(f'({self.epoch+1:03}/{self.epochs}) recall:{self.recalls[self.epoch]:>6.2f} f1:{self.f1s[self.epoch]:>6.4f} specification:{self.specs[self.epoch]:>5.2f}%', flush=True)

if __name__ == '__main__':
    from tqdm import tqdm
    args = utils.parse_args()
    torch.manual_seed(args.seed)
    trainer = SupervisedTrainer(args)
    for epoch in tqdm(range(trainer.epochs)):
        trainer.train()
        trainer.test()
        trainer.print_train_info()
        if (trainer.epoch+1)%10 == 0:
            trainer.save_model()
        trainer.epoch += 1