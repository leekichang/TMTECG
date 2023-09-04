"""
Created on Thu Aug 17 2023
@author: Kichang Lee
@contact: kichan.lee@yonsei.ac.kr
"""
import os
import torch
import numpy as np
import config as cfg
from tqdm import tqdm
from torchvision import transforms
from torch.utils.data import Dataset
from concurrent.futures import ThreadPoolExecutor

def minmax_scaling(data, new_min=0, new_max=1):
    data_min = np.min(data, axis=1, keepdims=True)
    data_max = np.max(data, axis=1, keepdims=True)
    scaled_data = new_min + (data - data_min) * (new_max - new_min) / (data_max - data_min)
    return scaled_data

class TMT(Dataset):
    def __init__(self, is_train, path='./dataset', data_types=cfg.DATA_TYPES['cad']):
        '''
        stage in [1, 2, 3, 4, #1, #2, #3, resting, SITTING]
        '''
        self.patient, self.data, self.labels  = [], [], []
        for t in data_types:
            path_ = f'{path}/{t}/train' if is_train else f'{path}/{t}/test'
            files = [file for file in os.listdir(path_) if file.endswith('.npz')]
            data_and_labels = self.load_data_parallel(files, path_)
            for data, label in data_and_labels:
                self.data.append(data)
                self.labels.append(label)
            # for file in tqdm(files):
            #     file = np.load(f'{path_}/{file}')
            #     self.data.append(file['data'])
            #     self.labels.append(file['label'])
        self.data   = torch.FloatTensor(np.array(self.data))
        self.labels = torch.LongTensor(np.array(self.labels))
        print(f'{"Train" if is_train else "Test"} data: {self.data.shape} target: {self.labels.shape}')
        
    def load_data(self, file, path):
        file_path = f'{path}/{file}'
        file_data = np.load(file_path)
        return file_data['data'], file_data['label']

    def load_data_parallel(self, files, path):
        with ThreadPoolExecutor(max_workers=12) as executor:
            data_and_labels = list(tqdm(executor.map(lambda file: self.load_data(file, path), files), total=len(files)))
        return data_and_labels
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        data_item  = self.data[idx]
        label_item = self.labels[idx]
        return data_item, label_item

class TMT_Full(Dataset):
    '''
    batch w.r.t the subjects?
    each subject has different number of samples
    main thread load N user's data and run the train phase
    other threads load another batch?
    '''
    def __init__(self, idx, is_train=None, path='./dataset', data_types=cfg.DATA_TYPES['cad']):
        path = path+'/full/train'
        self.subjects = [os.path.join(path, file) for file in os.listdir(path) if file.endswith('.npz')]
        self.data = self.load_data_parallel(self.subjects)
        print(np.shape(self.data))
    
    def load_data(self, file):
        file_data = np.load(file)
        return file_data['data']

    def load_data_parallel(self, files):
        with ThreadPoolExecutor(max_workers=12) as executor:
            data = list(tqdm(executor.map(lambda file: self.load_data(file), files), total=len(files)))
        return data

    def __len__(self):
        pass
        #return len(self.data)
    
    def __getitem__(self, idx):
        pass
        # data_item  = self.data[idx,:]
        # return data_item

if __name__ == '__main__':
    from tqdm import tqdm
    from torch.utils.data import DataLoader
    import utils
    
    args         = utils.parse_args()
    args.dataset = 'full'
    dataset      = utils.load_dataset(args)
    # print(1-torch.sum(dataset.labels)/len(dataset), torch.sum(dataset.labels)/len(dataset))
    # dataset      = utils.load_dataset(args, is_train=False)
    # dataloader   = DataLoader(dataset, batch_size=args.batch_size, shuffle=True , drop_last=True )
    
    # model = utils.build_model(args)
    # model = model.to('cuda')
    
    # for data, target in tqdm(dataloader):
    #     print(data.shape, target.shape)
    #     break
    #     data, target = data.to('cuda'), target.to('cuda')
    #     out = model(data)
    #     break
    # print(data.shape, out.shape)