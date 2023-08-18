"""
Created on Thu Aug 17 2023
@author: Kichang Lee
@contact: kichan.lee@yonsei.ac.kr
"""
import torch
import numpy as np
from torchvision import transforms
from torch.utils.data import Dataset

def minmax_scaling(data, new_min=0, new_max=1):
    data_min = np.min(data, axis=1, keepdims=True)
    data_max = np.max(data, axis=1, keepdims=True)
    scaled_data = new_min + (data - data_min) * (new_max - new_min) / (data_max - data_min)
    return scaled_data

class TMT(Dataset):
    def __init__(self, stage, is_train, path='./dataset/TMT_labeled'):
        is_train = 'train' if is_train else 'test'
        if stage != 'all':
            self.data   = np.load(f'{path}/STAGE{stage}_X_{is_train}.npy').transpose(0,2,1)
            self.data   = torch.FloatTensor(self.data)
            self.labels = torch.LongTensor(np.load(f'{path}/STAGE{stage}_Y_{is_train}.npy'))
        else:
            self.data, self.labels = [], []
            for i in range(4):
                chunk_X = np.load(f'{path}/STAGE{i+1}_X_{is_train}.npy').transpose(0,2,1)
                chunk_Y = np.load(f'{path}/STAGE{i+1}_Y_{is_train}.npy')
                self.data.append(chunk_X)
                self.labels.append(chunk_Y)
            self.data = torch.FloatTensor(np.concatenate(self.data, axis=0))
            self.labels = torch.LongTensor(np.concatenate(self.labels, axis=0))
            print(self.data.shape, self.labels.shape)
            
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        data_item  = self.data[idx,:]
        label_item = self.labels[idx]
        return data_item, label_item

class TMT_Full(Dataset):
    def __init__(self, idx, is_train=None, path='./dataset/TMT_unlabeled'):
        self.npz       = np.load(f'{path}/BATCH{idx}.npz')
        self.data      = self.npz['data'].transpose(0,2,1)
        print(f"{len(self.npz['count'])} Patients!, Total {sum(self.npz['count'])} Sample in Shape {self.data.shape}")
        self.data      = torch.FloatTensor(self.data)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        data_item  = self.data[idx,:]
        return data_item

if __name__ == '__main__':
    from tqdm import tqdm
    from torch.utils.data import DataLoader
    import utils
    
    args       = utils.parse_args()
    stage      = 'STAGE1'
    dataset    = utils.load_dataset(args, is_train=True)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True , drop_last=True )
    
    model = utils.build_model(args)
    model = model.to('cuda')
    
    for data, target in tqdm(dataloader):
        print(data.shape, target.shape)
        data, target = data.to('cuda'), target.to('cuda')
        out = model(data)
        break
    print(data.shape, out.shape)