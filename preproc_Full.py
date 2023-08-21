"""
Created on Thu Aug 21 2023
@author: Kichang Lee
@contact: kichan.lee@yonsei.ac.kr
"""
import os
import time
import numpy as np
import pandas as pd 
from tqdm import tqdm
import multiprocessing
from scipy.signal import resample
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from sklearn.model_selection import train_test_split

'''
_s_list.csv: TMT stage info
_s_*.csv   : csv file contains TMT stage data   (5000, 12) | 500Hz
_full.csv  : csv file contains Full TMT data    (N, 12) | 200Hz -> each patient have different N

# of patient : 16232
# of patient with CAD: 1632
'''
DATAPATH = f'../dr-you-ecg-20220420_mount/DachungBoo_TMT/dhkim2'
SAVEPATH = f'./dataset/TMT_labeled_Whole'
os.makedirs(SAVEPATH, exist_ok=True)

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
    database = pd.read_csv(f'../dr-you-ecg-20220420_mount/DachungBoo_TMT/20230821_TMT_WHOLE_DATA.csv', encoding='cp949')
    database['CAD_OUTCOME'] = database['CAD_OUTCOME'].fillna(0)
    fnames = [name.split('/')[-1] for name in database['fname'].to_list()]
    labels = [int(label) for label in database['CAD_OUTCOME'].to_list()]
    train_files, val_files, train_labels, val_labels = train_test_split(fnames, labels, test_size=0.2, random_state=42)
    train_counts = [train_labels.count(i) for i in range(2)]
    val_counts   = [  val_labels.count(i) for i in range(2)]
    print(sum(train_counts), train_counts[0], train_counts[1], train_counts[0]/sum(train_counts), train_counts[1]/sum(train_counts))
    print(  sum(val_counts),   val_counts[0],   val_counts[1],     val_counts[0]/sum(val_counts),     val_counts[1]/sum(val_counts))
    print(sum(train_counts)/(sum(train_counts)+sum(val_counts)))

    #desired_stages = [f'STAGE {i+1}' for i in range(4)]
    
    desired_stages = ['STAGE 1',
                      'STAGE 2',
                      'STAGE 3',
                      'STAGE 4',
                      'SITTING',
                      '#1',
                      '#2',
                      '#3',
                      ]
    
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
                if isinstance(stage, str) and 'STAGE' in stage:
                    train_file_dict[ 'all'].append(f'{f}_s_i{stripIndex}.csv')
                    train_label_dict['all'].append(train_labels[idx])
                    
                if isinstance(stage, str) and '#' in stage:
                    train_file_dict[ 'STAGEresting'].append(f'{f}_s_i{stripIndex}.csv')
                    train_label_dict['STAGEresting'].append(train_labels[idx])
                    
                for stage_ in desired_stages:
                    if isinstance(stage, str) and stage_ in stage:
                        key_name = f'STAGE{stage_}' if ('#' in stage_ or stage_=='SITTING') else stage_
                        train_file_dict[ key_name].append(f'{f}_s_i{stripIndex}.csv')
                        train_label_dict[key_name].append(train_labels[idx])
    
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
                
                if isinstance(stage, str) and 'STAGE' in stage:
                    test_file_dict[ 'all'].append(f'{f}_s_i{stripIndex}.csv')
                    test_label_dict['all'].append(val_labels[idx])
                    
                if isinstance(stage, str) and '#' in stage:
                    test_file_dict[ 'STAGEresting'].append(f'{f}_s_i{stripIndex}.csv')
                    test_label_dict['STAGEresting'].append(val_labels[idx])
                        
                for stage_ in desired_stages:
                    if isinstance(stage, str) and stage_ in stage:
                        key_name = f'STAGE{stage_}' if ('#' in stage_ or stage_=='SITTING') else stage_
                        test_file_dict[ key_name].append(f'{f}_s_i{stripIndex}.csv')
                        test_label_dict[key_name].append(val_labels[idx])
                    
    num_processes = multiprocessing.cpu_count()
    
    for stage in train_file_dict.keys():
        train_set = np.array([train_file_dict[stage], train_label_dict[stage]]).transpose(1,0)
        datas, targets = [], []
        with multiprocessing.Pool(processes=num_processes) as pool:
            results = list(tqdm(pool.imap(process_file, train_set), total=len(train_set)))
        for data, target in results:
            datas.append(data)
            targets.append(target)
        
        os.makedirs(SAVEPATH, exist_ok=True)
        np.savez_compressed(f"{SAVEPATH}/{stage.replace(' ', '')}_train.npz", data=np.array(datas), target=np.array(targets))
        # np.save(f"{SAVEPATH}/{stage.replace(' ', '')}_X_train.npy", np.array(datas))
        # np.save(f"{SAVEPATH}/{stage.replace(' ', '')}_Y_train.npy", np.array(targets))
        print(f"{stage} trainset {np.shape(datas)} saved!")
        
        test_set = np.array([test_file_dict[stage], test_label_dict[stage]]).transpose(1,0)
        datas, targets = [], []
        with multiprocessing.Pool(processes=num_processes) as pool:
            results = list(tqdm(pool.imap(process_file, test_set), total=len(test_set)))
        
        for data, target in results:
            datas.append(data)
            targets.append(target)
        
        os.makedirs(SAVEPATH, exist_ok=True)
        start_t = time.time()
        np.savez_compressed(f"{SAVEPATH}/{stage.replace(' ', '')}_test.npz", data=np.array(datas), target=np.array(targets))
        end_t = time.time()
        # np.save(f"{SAVEPATH}/{stage.replace(' ', '')}_X_test.npy", np.array(datas))
        # np.save(f"{SAVEPATH}/{stage.replace(' ', '')}_Y_test.npy", np.array(targets))
        print(f"{stage} testset {np.shape(datas)} saved! Took {end_t-start_t:.2f} sec!")
        # else:
        #     pass
        #     #TODO: ALL data 관련해서 학습데이터셋 정리!!
        #     #TEST SET
        #     num_batch   = 16
        #     batch_size  = int(len(test_file_dict[stage])//num_batch)
        #     batch_idx   = [i*batch_size for i in range(num_batch)]
        #     batch_idx.append(len(test_file_dict[stage]))
        #     num_threads = 16
        #     for idx in tqdm(range(len(batch_idx)-1)):
        #         test_set = np.array([test_file_dict[stage][batch_idx[idx]:batch_idx[idx+1]],
        #                             test_label_dict[stage][batch_idx[idx]:batch_idx[idx+1]]]).transpose(1,0)
        #         datas, targets = [], []
        #         print(test_set.shape)
        #         with ThreadPoolExecutor(max_workers=num_threads) as executor:
        #             batch_results = list(tqdm(executor.map(process_file, test_set), total=len(test_set)))
                
        #         for data, target in batch_results:
        #             if data is not None:
        #                 datas.append(data)
        #                 targets.append(target)
        #         datas = np.array(datas, dtype=np.float32)
        #         targets = np.array(targets, dtype=np.int32)
        #         print(f"====== BATCH {idx+1} ======")
        #         print("Final data shape:", datas.shape)
        #         print("Final targets shape:", targets.shape)
        #         start_t = time.time()
        #         np.savez_compressed(f"{SAVEPATH}/{stage.replace(' ', '')}_{idx+1}_test.npz", data=datas, target=targets)
        #         end_t = time.time()
        #         print(f" saved!")
        #         print(f'{stage} testset BATCH {idx+1} {np.shape(datas)} Saved! Took {end_t-start_t:.2f} sec!')

        #     # #TRAIN SET
            
        #     num_threads = 16
        #     for idx in tqdm(range(len(batch_idx)-1)):
        #         train_set = np.array([train_file_dict[stage][batch_idx[idx]:batch_idx[idx+1]],
        #                             train_file_dict[stage][batch_idx[idx]:batch_idx[idx+1]]]).transpose(1,0)
        #         datas, targets = [], []
        #         print(test_set.shape)
        #         with ThreadPoolExecutor(max_workers=num_threads) as executor:
        #             batch_results = list(tqdm(executor.map(process_file, test_set), total=len(train_set)))
                
        #         for data, target in batch_results:
        #             if data is not None:
        #                 datas.append(data)
        #                 targets.append(target)
                        
        #         datas = np.array(datas, dtype=np.float32)
        #         targets = np.array(targets, dtype=np.int32)
        #         print(f"====== BATCH {idx+1} ======")
        #         print("Final data shape:", datas.shape)
        #         print("Final targets shape:", targets.shape)
        #         start_t = time.time()
        #         np.savez_compressed(f"{SAVEPATH}/{stage.replace(' ', '')}_{idx+1}_train.npz", data=datas, target=targets)
        #         end_t = time.time()
        #         print(f" saved!")
        #         print(f'{stage} trainset BATCH {idx+1} {np.shape(datas)} Saved! Took {end_t-start_t:.2f} sec!')
    