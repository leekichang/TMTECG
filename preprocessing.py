"""
Created on Thu Aug 17 2023
@author: Kichang Lee
@contact: kichan.lee@yonsei.ac.kr
"""
import os
import numpy as np
import pandas as pd 
from tqdm import tqdm
import multiprocessing
from scipy.signal import resample
from collections import defaultdict
from sklearn.model_selection import train_test_split

'''
_s_list.csv: TMT stage info
_s_*.csv   : csv file contains TMT stage data   (5000, 12) | 500Hz
_full.csv  : csv file contains Full TMT data    (N, 12) | 200Hz -> each patient have different N

# of patient : 16232
# of patient with CAD: 1632
'''
DATAPATH = f'../dr-you-ecg-20220420_mount/DachungBoo_TMT/dhkim2'
SAVEPATH = f'./dataset/TMT_labeled'

def process_file(files):
    source_sr, target_sr = 500, 250  # for snapshot 500Hz / for full 200Hz -> 250 Hz
    file_name = files[0]
    target = int(files[1])
    df = pd.read_csv(f'{DATAPATH}/{file_name}')
    data = df.to_numpy()
    data = resample_data(data, source_sr, target_sr)
    return data, target

def resample_data(data, original_rate, target_rate):
    original_length = data.shape[0]
    target_length = int(original_length * target_rate / original_rate)
    resampled_data = resample(data, target_length)
    return resampled_data

if __name__ == '__main__':
    database = pd.read_csv(f'use_this_data/20230805_CAD_DF_FINAL.csv')
    fnames = [name.split('/')[-1] for name in database['fname'].to_list()]
    labels = [int(label) for label in database['CAD'].to_list()]
    train_files, val_files, train_labels, val_labels = train_test_split(fnames, labels, test_size=0.2, random_state=42)
    
    desired_stages = [f'STAGE {i+1}' for i in range(4)]
    
    train_file_dict  = defaultdict(list)
    train_label_dict = defaultdict(list)

    '''
    Train Set
    '''
    for idx, f in enumerate(tqdm(train_files)):
        if not os.path.exists(f"{DATAPATH}/{f}_s_list.csv"):
            print(f"*** {f} DOESN'T HAVE STAGE INFORMATION! ***", flush=True)
            
        else:
            df = pd.read_csv(f"{DATAPATH}/{f}_s_list.csv")
            for jdx in range(len(df)):
                stage      = df.loc[jdx]['StageName']
                stripIndex = df.loc[jdx]['StripIndex']
                
                if stage in desired_stages:
                    train_file_dict[stage].append(f'{f}_s_i{stripIndex}.csv')
                    train_label_dict[stage].append(train_labels[idx])
    
    test_file_dict  = defaultdict(list)
    test_label_dict = defaultdict(list)

    '''
    Test Set
    '''
    for idx, f in enumerate(tqdm(val_files)):
        if not os.path.exists(f"{DATAPATH}/{f}_s_list.csv"):
            print(f"*** {f} DOESN'T HAVE STAGE INFORMATION! ***", flush=True)
            
        else:
            df = pd.read_csv(f"{DATAPATH}/{f}_s_list.csv")
            for jdx in range(len(df)):
                stage      = df.loc[jdx]['StageName']
                stripIndex = df.loc[jdx]['StripIndex']
                
                if stage in desired_stages:
                    test_file_dict[stage].append(f'{f}_s_i{stripIndex}.csv')
                    test_label_dict[stage].append(val_labels[idx])
    
    num_processes = multiprocessing.cpu_count()
    
    for stage in desired_stages:
        train_set = np.array([train_file_dict[stage], train_label_dict[stage]]).transpose(1,0)
        datas, targets = [], []
        with multiprocessing.Pool(processes=num_processes) as pool:
            results = list(tqdm(pool.imap(process_file, train_set), total=len(train_set)))
        
        for data, target in results:
            datas.append(data)
            targets.append(target)
        
        os.makedirs(SAVEPATH, exist_ok=True)
        
        np.save(f"{SAVEPATH}/{stage.replace(' ', '')}_X_train.npy", np.array(datas))
        np.save(f"{SAVEPATH}/{stage.replace(' ', '')}_Y_train.npy", np.array(targets))
        print(f"{stage} trainset {np.shape(datas)} saved!")
        
        test_set = np.array([test_file_dict[stage], test_label_dict[stage]]).transpose(1,0)
        datas, targets = [], []
        with multiprocessing.Pool(processes=num_processes) as pool:
            results = list(tqdm(pool.imap(process_file, test_set), total=len(test_set)))
        
        for data, target in results:
            datas.append(data)
            targets.append(target)
        
        os.makedirs(SAVEPATH, exist_ok=True)
        
        np.save(f"{SAVEPATH}/{stage.replace(' ', '')}_X_test.npy", np.array(datas))
        np.save(f"{SAVEPATH}/{stage.replace(' ', '')}_Y_test.npy", np.array(targets))
        print(f"{stage} testset {np.shape(datas)} saved!")    
    