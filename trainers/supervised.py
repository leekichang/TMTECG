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
from sklearn.metrics import accuracy_score, recall_score, f1_score, balanced_accuracy_score, roc_auc_score, roc_curve, confusion_matrix, auc

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
        args.dataset      = args.trainset
        self.trainset     = utils.load_dataset(args, is_train=True)
        class_weight      = torch.sum(self.trainset.labels)/len(self.trainset)
        self.criterion    = utils.build_criterion(args, class_weight).to(self.device)
        args.dataset      = args.testset
        self.testset      = utils.load_dataset(args, is_train=False)
        self.model        = utils.build_model(args).to(self.device)
        self.optimizer    = utils.build_optimizer(self.model, args)
        
        self.train_loader = DataLoader(self.trainset, batch_size=args.batch_size, shuffle=True , drop_last=True )
        self.test_loader  = DataLoader(self.testset , batch_size=args.batch_size, shuffle=False, drop_last=False)
    
        self.train_loss = None
        self.test_loss  = None
        self.acc        = None
        self.sens       = None
        self.f1         = None
        self.spec       = None
        self.bal_acc    = None
        self.auroc      = None
        
        self.TB_WRITER = tb.SummaryWriter(f'./tensorboard/{str(datetime.now().strftime("%Y-%m-%d_%H:%M:%S"))}_{self.args.exp_name}') \
            if self.args.use_tb else None
        
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f'model name:{args.model}\ndataset:{args.dataset}\ndevice:{self.device}\nTensorboard:{self.args.use_tb}\nTotal parameter:{total_params:,}')
        

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
        self.train_loss = np.mean(losses)
        
        if self.args.use_tb:
            self.TB_WRITER.add_scalar("Train Loss", self.train_loss, self.epoch+1)
        
    @torch.no_grad()
    def test(self):
        self.model.eval()
        probs, targets, losses = [], [], []
        for X, Y in self.test_loader:
            X, Y = X.to(self.device), Y.to(self.device)
            pred = self.model(X)
            loss = self.criterion(pred, Y)
            prob = torch.softmax(pred, dim=1)
            
            probs.append(prob.cpu().numpy())
            targets.append(Y.cpu().numpy())
            losses.append(loss.item())
        
        probs = np.concatenate(probs)
        targets = np.concatenate(targets)
        
        
        
        fpr, tpr, thresholds = roc_curve(targets, probs[:, 1])
        sensitivity = tpr
        specificity = 1 - fpr
        j_index = sensitivity + specificity - 1
        best_threshold_idx = np.argmax(j_index)
        best_threshold = thresholds[best_threshold_idx]
        binary_predictions = (probs[:, 1] > best_threshold).astype(int)
        
        acc     = accuracy_score(y_true=targets, y_pred=binary_predictions)
        sens    = recall_score(y_true=targets, y_pred=binary_predictions)
        f1      = f1_score(y_true=targets, y_pred=binary_predictions)
        spec    = utils.specificity_score(y_true=targets, y_pred=binary_predictions)
        bal_acc = balanced_accuracy_score(y_true=targets, y_pred=binary_predictions)
        auroc   = roc_auc_score(y_true=targets, y_score=probs[:, 1])
        
        self.test_loss = np.mean(losses)
        self.acc       = acc
        self.sens      = sens
        self.f1        = f1
        self.spec      = spec
        self.bal_acc   = bal_acc
        self.auroc     = auroc
        
        if self.args.use_tb:
            self.TB_WRITER.add_scalar(f'Test Loss', self.test_loss, self.epoch+1)
            self.TB_WRITER.add_scalar(f'Test Accuracy', self.acc, self.epoch+1)
            self.TB_WRITER.add_scalar(f'Sensitivity', sens, self.epoch+1)
            self.TB_WRITER.add_scalar(f'F1-Score', f1, self.epoch+1)
            self.TB_WRITER.add_scalar(f'Specificity', spec, self.epoch+1)
            self.TB_WRITER.add_scalar(f'Test Accuracy (Balanced)', bal_acc, self.epoch+1)
            self.TB_WRITER.add_scalar(f'AUROC', auroc, self.epoch+1)
    
    def save_model(self):
        torch.save(self.model.state_dict(), f'{self.save_path}/{self.epoch+1}.pth')

    def print_train_info(self):
        print(f'({self.epoch+1:03}/{self.epochs}) Train Loss:{self.train_loss:>6.4f} Test Loss:{self.test_loss:>6.4f} Test Accuracy:{self.acc*100:>5.2f}% Balanced Test Accuracy:{self.bal_acc*100:>5.2f}% Sensitivity:{self.sens:>6.4f} f1:{self.f1:>6.4f} specificity:{self.spec:>5.4f} AUROC:{self.auroc:>5.4f}', flush=True)

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