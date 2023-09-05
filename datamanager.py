"""
Created on Thu Aug 17 2023
@author: Kichang Lee
@contact: kichan.lee@yonsei.ac.kr
"""
import os
import copy
import torch
import queue
import random
import threading
import numpy as np
import config as cfg
from tqdm import tqdm
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import Dataset
import multiprocessing
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
        self.data       = F.normalize(torch.FloatTensor(np.array(self.data)), dim=-1)
        self.labels     = torch.LongTensor(np.array(self.labels))
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
    def __init__(self, is_train=None, path='./dataset', data_types=cfg.DATA_TYPES['cad']):
        path = path+'/full/train'
        self.subjects_  = [os.path.join(path, file) for file in os.listdir(path) if file.endswith('.npz')]
        print(len(self.subjects_))
        self.subjects_  = self.subjects_[15000:]
        self.subjects   = None
        
        self.chunk_size = 100
        self.chunk_idx  = 0
        self.num_chunks = len(self.subjects_) // self.chunk_size + 1
        self.data       = None # self.load_data_parallel(self.subjects[self.chunk_idx])
        self.next_data  = None

        self.data_queue = queue.Queue()
        
    def setup(self):
        self.shuffle_subject()        
        self.split_subject()
        if self.next_data == None:
            self.data = self.load_data_parallel(self.subjects[self.chunk_idx])
            print(self.data.shape)
        else:
            self.update()
    
    def shuffle_subject(self):
        random.shuffle(self.subjects_)

    def split_subject(self):
        self.subjects = [self.subjects_[i:i + self.chunk_size] for i in range(0, len(self.subjects_), self.chunk_size)]

    def load_data(self, file):
        file_data = np.load(file)
        return file_data['data']
    
    def load_next_chunk(self):
        if self.chunk_idx < self.num_chunks:
            self.next_data = self.load_data_parallel(self.subjects[self.chunk_idx+1])
        else:
            self.chunk_idx = -1
            self.next_data = self.load_data_parallel(self.subjects[self.chunk_idx+1])
            
    def update(self):
        self.chunk_idx += 1
        self.data = self.next_data
    
    def load_data_parallel(self, files):
        with ThreadPoolExecutor(max_workers=4) as executor:
            data = list(executor.map(lambda file: self.load_data(file), files))# list(tqdm(executor.map(lambda file: self.load_data(file), files), total=len(files)))
        data = torch.FloatTensor(np.concatenate(data))
        return data

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

def load_data(file):
    file_data = np.load(file)
    return file_data['data']

def load_data_parallel(files):
    with ThreadPoolExecutor(max_workers=12) as executor:
        data = list(executor.map(lambda file: load_data(file), files))# list(tqdm(executor.map(lambda file: self.load_data(file), files), total=len(files)))
    data = torch.FloatTensor(np.concatenate(data))
    return data

def load_next_chunk(chunk_idx, num_chunks, files):
    if chunk_idx < num_chunks:
        next_data = load_data_parallel(files) # subjects[chunk_idx+1])
    else:
        chunk_idx = -1
        next_data = load_data_parallel(files) # subjects[chunk_idx+1])
    return chunk_idx, next_data

def background_loading(chunk_idx, num_chunks, files, data_queue):
    chunk_idx, next_data = load_next_chunk(chunk_idx, num_chunks, files)
    print(f'{chunk_idx+1} data ready! SHAPE: {next_data.shape}')
    return data_queue.put((chunk_idx, next_data))

if __name__ == '__main__':
    from tqdm import tqdm
    from torch.utils.data import DataLoader
    import utils
    import time
    
    args         = utils.parse_args()
    args.dataset = 'full'
    dataset      = utils.load_dataset(args)
    
    
    
    data_queue = multiprocessing.Queue()
    
    
    for epoch in range(2):
        dataset.setup()
        for chunk_idx in tqdm(range(dataset.num_chunks)):
            data_queue = queue.Queue()
            completion_event = threading.Event()
            background_thread = threading.Thread(target=background_loading, args=(dataset.chunk_idx, dataset.num_chunks, dataset.subjects[chunk_idx], data_queue))
            background_thread.daemon = True
            background_thread.start()
            # background_thread.join()
            dataloader   = DataLoader(dataset, batch_size=128, shuffle=False, drop_last=False)
            for idx, data in enumerate(dataloader):
                if idx % 10 == 0:
                    print(idx, dataset.chunk_idx, data.shape)
                    time.sleep(0.25)
            chunk_idx, next_data = data_queue.get()
            dataset.chunk_idx = chunk_idx
            dataset.next_data = next_data
            dataset.update()
        
        
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