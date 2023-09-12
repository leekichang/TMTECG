"""
Created on Thu Aug 17 2023
@author: Kichang Lee
@contact: kichan.lee@yonsei.ac.kr
"""
import torch
import argparse
import numpy as np
import torch.nn as nn
import torch.optim as optim
from concurrent.futures import ThreadPoolExecutor

import models
import trainers
import datamanager
import config as cfg
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', help='experiement name', type=str, default='CNN_B_CMSC')
    parser.add_argument('--model', help='Model'  , type=str, default='CNN_B'  , choices=['CNN_B', 'CNN_Bg', 'CNN_single-B'])
    parser.add_argument('--dataset', help='Dataset', type=str, default='cad', choices=['angio', 'cad', 'whole', 'full'])
    parser.add_argument('--trainset', type=str, default='cad', choices=['angio', 'cad', 'whole', 'full'])
    parser.add_argument('--testset', type=str, default='cad', choices=['angio', 'cad', 'whole', 'full'])
    parser.add_argument('--phase', help='Phase', type=str, default='randominit', choices=['finetune', 'linear', 'randominit', 'SimCLR', 'BYOL'])
    parser.add_argument('--loss', help='Loss function', type=str, default='CrossEntropyLoss')
    parser.add_argument('--optimizer', help='Optimizer', type=str, default='AdamW')
    parser.add_argument('--lr', help='Learning rate', type=float, default=0.001)
    parser.add_argument('--decay', help='Weight decay', type=float, default=0.001)
    parser.add_argument('--batch_size', help='Batch size', type=int, default=128)
    parser.add_argument('--epochs', help='Epochs', type=int, default=100)
    parser.add_argument('--ckpt_freq', type=int, default=20)
    parser.add_argument('--seed', type=int, default=0)
     
    parser.add_argument('--t', help='temperature for SimCLR', type=float, default=0.5)
    parser.add_argument('--ma_decay', help='Moving average decay', type=float, default=0.9)
    
    parser.add_argument('--datapath', type=str, default='./dataset')
    parser.add_argument('--test_batch', type=int, default=2048)
    parser.add_argument('--ckpt_path', type=str, default='SimCLR_4096_aug_pretrain')
    parser.add_argument('--ckpt_epoch', type=int, default=3)
    
    parser.add_argument('--use_tb', type=str2bool, default=False)

    args = parser.parse_args()
    return args

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def load_dataset(args, is_train=True):
    data_type    = args.trainset if is_train else args.testset    
    dataset_type = 'TMT' if args.phase in ['finetune', 'linear', 'randominit'] else 'TMT_Full'
    return getattr(datamanager, dataset_type)(is_train, path=args.datapath, data_types=cfg.DATA_TYPES[data_type])

def build_model(args):
    model, model_type = args.model.split('_')
    return getattr(models, model)(model_type=model_type, num_class=cfg.N_CLASS[args.dataset])

def build_criterion(args, class_weight=0.5):
    class_weights = torch.tensor([1-class_weight, class_weight])
    print(f"Weighted Loss) Negative: {class_weights[0]:.2f} Positive: {class_weights[1]:.2f}")
    return getattr(nn, args.loss)(weight = class_weights)

def build_optimizer(model, args):
    return getattr(optim, args.optimizer)(model.parameters(), lr=args.lr, weight_decay=args.decay)

def build_trainer(args):
    trainer_type = cfg.TRAINER[args.phase]
    return getattr(trainers, trainer_type)(args)

def calculate_topk_accuracy(predictions, targets, k):
    _, topk_preds = predictions.topk(k, dim=1)
    correct = topk_preds.eq(targets.view(-1, 1).expand_as(topk_preds))
    topk_acc = correct.any(dim=1).float().mean().item() * 100
    return topk_acc

def specificity_score(y_pred, y_true):
    true_negative = np.sum(np.logical_and(y_pred == 0, y_true == 0))
    actual_negative = np.sum(y_true == 0)
    
    specificity = true_negative / (actual_negative + 1e-7)
    return specificity

def set_requires_grad(model, val):
    for p in model.parameters():
        p.requires_grad = val
        
def load_data(file):
    file_data = np.load(file)
    return file_data['data']

def load_data_parallel(files):
    with ThreadPoolExecutor(max_workers=12) as executor:
        data = list(executor.map(lambda file: load_data(file), files)) # list(tqdm(executor.map(lambda file: self.load_data(file), files), total=len(files)))
    data = torch.FloatTensor(np.concatenate(data)) if 'full' in files[0] else torch.FloatTensor(np.array(data))
    return data

def load_next_chunk(chunk_idx, num_chunks, files):
    if chunk_idx < num_chunks:
        next_data = load_data_parallel(files)
    else:
        chunk_idx = -1
        next_data = load_data_parallel(files)
    return chunk_idx, next_data

def background_loading(chunk_idx, num_chunks, files, data_queue):
    chunk_idx, next_data = load_next_chunk(chunk_idx, num_chunks, files)
    print(f'{chunk_idx+1} data ready! SHAPE: {next_data.shape}', flush=True)
    return data_queue.put((chunk_idx, next_data))