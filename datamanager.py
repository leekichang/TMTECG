"""
Created on Thu Aug 17 2023
@author: Kichang Lee
@contact: kichan.lee@yonsei.ac.kr
"""
import os
import torch
import queue
import random
import pickle
import threading
import numpy as np
import config as cfg
from tqdm import tqdm
from functools import partial
import torch.nn.functional as F
from torch.utils.data import Dataset
from concurrent.futures import ThreadPoolExecutor

def minmax_scaling(data, new_min=0, new_max=1):
    data_min = np.min(data, axis=1, keepdims=True)
    data_max = np.max(data, axis=1, keepdims=True)
    scaled_data = new_min + (data - data_min) * (new_max - new_min) / (data_max - data_min)
    return scaled_data

with open('./dataset/full/subject_ids.pickle', 'rb') as f:
    SUBJECT_IDS = pickle.load(f)

class TMT(Dataset):
    def __init__(self, args, is_train, path='./dataset', data_types=cfg.DATA_TYPES['cad']):
        self.args = args
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
    def __init__(self, args, is_train=None, path='./dataset', data_types=cfg.DATA_TYPES['full']):
        self.args = args
        self.subjects_ = []
        self.data_types = data_types
        for t in self.data_types:
            path_ = f'{path}/{t}/train'
            files = [os.path.join(path_, file) for file in os.listdir(path_) if file.endswith('.npz')]
            self.subjects_ = self.subjects_ + files
        
        self.subjects   = None
        
        self.chunk_size = 250
        self.chunk_idx  = 0
        self.num_chunks = len(self.subjects_)//self.chunk_size + (0 if len(self.subjects_)%self.chunk_size==0 else 1)
        self.data       = None # self.load_data_parallel(self.subjects[self.chunk_idx])
        self.next_data  = None

        self.data_queue = queue.Queue()
        
        self.data_shape = (-1,12,5000) if self.args.phase in ['CMSC'] else (-1,12,2500)
        
    def setup(self):
        self.shuffle_subject()        
        self.split_subject()
        if self.next_data == None:
            self.data = load_data_parallel(self.subjects[self.chunk_idx], shape=self.data_shape)
        else:
            self.update()
    
    def shuffle_subject(self):
        random.shuffle(self.subjects_)

    def split_subject(self):
        self.subjects = [self.subjects_[i:i + self.chunk_size] for i in range(0, len(self.subjects_), self.chunk_size)]

    def update(self):
        self.chunk_idx += 1
        self.data = self.next_data
        
    # def load_data(self, file):
    #     file_data = np.load(file)
    #     return file_data['data']
    
    # def load_next_chunk(self):
    #     if self.chunk_idx < self.num_chunks:
    #         self.next_data = self.load_data_parallel(self.subjects[self.chunk_idx+1])
    #     else:
    #         self.chunk_idx = -1
    #         self.next_data = self.load_data_parallel(self.subjects[self.chunk_idx+1])
            
    
    # def load_data_parallel(self, files):
    #     if len(files) > 0:
    #         with ThreadPoolExecutor(max_workers=4) as executor:
    #             data = list(executor.map(lambda file: self.load_data(file), files))
    #         data = torch.FloatTensor(np.concatenate(data)) if 'full' in files[0] else torch.FloatTensor(np.array(data))
    #         return data

    def __len__(self):
        return len(self.data['data'])
    
    def __getitem__(self, idx):
        data   = self.data['data'][idx]
        id_    = self.data['id'][idx]
        length = self.data['length'][idx]
        return data, id_, length

def load_data(file, shape=(-1,12,5000)):
    file_data = np.load(file)
    data = file_data['data']
    # print(file_data['id'])
    if str(file_data['id']) in list(SUBJECT_IDS.keys()):
        id_  = SUBJECT_IDS[str(file_data['id'])]
    else:
        print(f"NO USER ID: {file_data['id']}")
        return 
    result = {'data':None, 'id':None, 'length':None}
    if shape[2] == 2500:
        result['data']   = data
        result['length'] = np.arange(data.shape[0])
        result['id']     = np.ones(data.shape[0])*id_
        return result
    elif shape[2] == 5000:
        B, _, _ = data.shape
        if B % 2 == 0:
            result['data']   = data.reshape(shape)
            result['length'] = np.arange(data.shape[0])
            result['id']     = np.ones(data.shape[0])*id_
            if result['data'] is None:
                print("NO DATA")
            return result
        else:
            result['data']   = data[:B-1].reshape(shape)
            result['length'] = np.arange(result['data'].shape[0])
            result['id']     = np.ones(result['data'].shape[0])*id_
            if result['data'] is None:
                print("NO DATA")
            return result

def load_data_parallel(files, shape):
    if len(files) > 0:
        with ThreadPoolExecutor(max_workers=12) as executor:
            # load_data 함수에 shape를 전달하기 위해 partial을 사용
            load_data_partial = partial(load_data, shape=shape)
            
            # map 함수를 사용하여 병렬로 데이터를 로드하고 결과를 수집
            results = list(executor.map(load_data_partial, files))

        # 결과에서 필요한 부분을 추출하여 새로운 데이터 구조 생성
        data = {
            'data'  : torch.FloatTensor(np.concatenate([result['data'] for result in results if result['data'] is not None])),
            'id'    : torch.LongTensor(np.concatenate([result['id'] for result in results])),
            'length': torch.LongTensor(np.concatenate([result['length'] for result in results]))
        }

        return data
    
def load_next_chunk(chunk_idx, num_chunks, files, shape):
    if chunk_idx < num_chunks:
        next_data = load_data_parallel(files, shape)
    else:
        chunk_idx = -1
        next_data = load_data_parallel(files, shape)
    return chunk_idx, next_data

def background_loading(chunk_idx, num_chunks, files, data_queue, shape):
    chunk_idx, next_data = load_next_chunk(chunk_idx, num_chunks, files, shape)
    print(f'{chunk_idx+1} data ready! SHAPE: {next_data["data"].shape}', flush=True)
    return data_queue.put((chunk_idx, next_data))

# def load_data(file, shape=(-1,12,5000)):
#     file_data = np.load(file)
#     data = file_data['data']
#     B, _, _ = data.shape
#     if B % 2 == 0:
#         return data.reshape(shape)
#     else:
#         return data[:B-1].reshape(shape)

# def load_data_parallel(files, shape):
#     if len(files) > 0:
#         with ThreadPoolExecutor(max_workers=12) as executor:
#             load_data_partial = partial(load_data, shape=shape)
#             data = list(executor.map(load_data_partial, files))
#             # data = list(executor.map(lambda file: load_data(file), files))
#         data = torch.FloatTensor(np.concatenate(data)) if 'full' in files[0] else torch.FloatTensor(np.array(data))
#         return data

# def load_next_chunk(chunk_idx, num_chunks, files, shape):
#     if chunk_idx < num_chunks:
#         next_data = load_data_parallel(files, shape)
#     else:
#         chunk_idx = -1
#         next_data = load_data_parallel(files, shape)
#     return chunk_idx, next_data

# def background_loading(chunk_idx, num_chunks, files, data_queue, shape):
#     chunk_idx, next_data = load_next_chunk(chunk_idx, num_chunks, files, shape)
#     # print(f'{chunk_idx+1} data ready! SHAPE: {next_data.shape}')
#     return data_queue.put((chunk_idx, next_data))

if __name__ == '__main__':
    from tqdm import tqdm
    from torch.utils.data import DataLoader
    import utils
    import time
    
    args          = utils.parse_args()
    args.dataset  = 'full'
    args.trainset = 'full'
    args.phase    = 'SimCLR'
    dataset       = utils.load_dataset(args)
    
    # data_queue = multiprocessing.Queue()
    shape = (-1, 12, 5000)
    for epoch in range(2):
        dataset.setup()
        for chunk_idx in tqdm(range(dataset.num_chunks)):
            data_queue = queue.Queue()
            completion_event = threading.Event()
            background_thread = threading.Thread(target=background_loading, args=(dataset.chunk_idx, dataset.num_chunks, dataset.subjects[chunk_idx], data_queue, shape))
            background_thread.daemon = True
            background_thread.start()
            dataloader   = DataLoader(dataset, batch_size=128, shuffle=False, drop_last=False)
            for idx, data in enumerate(dataloader):
                if idx % 10 == 0:
                    print(idx, dataset.chunk_idx, data.shape)
            chunk_idx, next_data = data_queue.get()
            dataset.chunk_idx = chunk_idx
            dataset.next_data = next_data
            dataset.update()
        