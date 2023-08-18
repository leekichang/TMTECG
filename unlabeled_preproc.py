"""
Created on Thu Aug 17 2023
@author: Kichang Lee
@contact: kichan.lee@yonsei.ac.kr
"""
import time
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.signal import resample
from concurrent.futures import ThreadPoolExecutor

DATAPATH = f'../dr-you-ecg-20220420_mount/DachungBoo_TMT/dhkim2'
SAVEPATH = f'./dataset/TMT_unlabeled'

def segment_data(data, sample_rate=250, window_size=10):
    window_samples = window_size * sample_rate
    num_windows = len(data) // window_samples
    num_samples = num_windows * window_samples
    data = data[:num_samples]
    split_data = np.split(data, num_windows)
    return np.array(split_data)

def resample_data(data, original_rate=200, target_rate=250):
    original_length = data.shape[0]
    target_length = int(original_length * target_rate / original_rate)
    resampled_data = resample(data, target_length)
    return resampled_data

def preprocess_file(file):
    df = pd.read_csv(f'{DATAPATH}/{file}')
    data = df.to_numpy()
    if data.shape[0] < 2000:
        return None, None
    data = resample_data(df.to_numpy())
    data = segment_data(data)
    return np.array(data), np.shape(data)[0]

if __name__ == '__main__':
    with open(file=f'{SAVEPATH}/full_files.pickle', mode='rb') as f:
        files = pickle.load(f)

    print(f"*** {len(files)} PATIENTS ***\n\n")

    num_batch   = 32
    batch_size  = int(len(files)//num_batch)
    batch_idx   = [i*batch_size for i in range(num_batch)]
    batch_idx.append(len(files))
    
    num_threads = 16
    for idx in tqdm(range(28, len(batch_idx)-1)):
        X, counts = [], []
        
        batch_files = files[batch_idx[idx]:batch_idx[idx+1]]
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            batch_results = list(tqdm(executor.map(preprocess_file, batch_files), total=len(batch_files)))
        
        for data, count in batch_results:
            if data is not None:
                X.append(data)
                counts.append(count)
                # print("File:", file, "Shape", data.shape, "Data count:", count)
        del batch_files
        del batch_results
        X = np.concatenate(X, axis=0, dtype=np.float32)
        counts = np.array(counts, dtype=np.int32)
        print(f"====== BATCH {idx+1} ======")
        print("Final X shape:", X.shape)
        print("Final Count shape:", counts.shape)
        start_t = time.time()
        np.savez_compressed(f'{SAVEPATH}/BATCH{idx+1}.npz', data=X, count=counts)
        end_t = time.time()
        print(f'BATCH {idx+1} Saved! Took {end_t-start_t:.2f} sec!')