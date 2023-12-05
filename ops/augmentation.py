"""
Created on Thu Aug 18 2023
@author: Kichang Lee
@contact: kichan.lee@yonsei.ac.kr
"""
import copy
import torch
import random
import torch.nn as nn
import ops.spec_aug as spec_aug

__all__ = [
    'Augmentator'
]

def x_flip(
    x:torch.FloatTensor
)->torch.FloatTensor:
    """
    x: (N,C,S) or (N, K, C, S)
    return: (N,C,S) or (N, K, C, S)
    
    Vertical (x-axis) Flip
    """
    return x*-1

def y_flip(
    x:torch.FloatTensor
)->torch.FloatTensor:
    """
    x: (N,C,S) or (N, K, C, S)
    return: (N,C,S) or (N, K, C, S)
    
    Horizontal (y-axis) Flip
    """
    return x.flip(dims=[-1])

def gaussian_noise(
    x:torch.FloatTensor,
    mean:float=0,
    std:float=0.1
)->torch.FloatTensor:
    """
    x: (N,C,S) or (N, K, C, S)
    return: (N,C,S) or (N, K, C, S)
    
    Add Guassian Noise
    """
    noise = (torch.randn(x.shape) * std + mean).to('cuda' if torch.cuda.is_available() else 'cpu')
    return x + noise

def spec_time(
    x:torch.FloatTensor,
    max_masked_bins,
    mask_size,
)->torch.FloatTensor:
    """
    x: (N,C,S) or (N, K, C, S)
    return: (N,C,S) or (N, K, C, S)
    
    Apply Spectral time masking
    """
    B, C, S = x.shape
    stft_data = torch.stft(x.view(-1, S), n_fft=256, hop_length=64, return_complex=True)
    stft_data = stft_data.view(B, C, stft_data.shape[-2], stft_data.shape[-1])
    masked_stft_data = spec_aug.random_time_masking(stft_data)
    
    istft_data = torch.istft(masked_stft_data.view(-1, masked_stft_data.shape[-2], masked_stft_data.shape[-1]), 
                            n_fft=256, hop_length=64, win_length=128)
    istft_data = istft_data.view(B, C, istft_data.shape[-1])  # 형태 조정

    _, _, S  = istft_data.shape
    istft_data_ = copy.deepcopy(x)
    istft_data_[:, :, :S] = istft_data
    return istft_data_

def spec_freq(
    x:torch.FloatTensor,
    max_masked_bins,
    mask_size,
)->torch.FloatTensor:
    """
    x: (N,C,S) or (N, K, C, S)
    return: (N,C,S) or (N, K, C, S)
    
    Apply Spectral frequency masking
    """
    B, C, S = x.shape
    stft_data = torch.stft(x.view(-1, S), n_fft=256, hop_length=64, return_complex=True)
    stft_data = stft_data.view(B, C, stft_data.shape[-2], stft_data.shape[-1])
    masked_stft_data = spec_aug.random_frequency_masking(stft_data)
    
    istft_data = torch.istft(masked_stft_data.view(-1, masked_stft_data.shape[-2], masked_stft_data.shape[-1]), 
                            n_fft=256, hop_length=64, win_length=128)
    istft_data = istft_data.view(B, C, istft_data.shape[-1])  # 형태 조정

    _, _, S  = istft_data.shape
    istft_data_ = copy.deepcopy(x)
    istft_data_[:, :, :S] = istft_data
    return istft_data_
    
class RandomApply(nn.Module):
    def __init__(self, fn, p):
        super().__init__()
        self.fn = fn
        self.p = p
    def forward(self, x):
        if isinstance(x, tuple):
            x, aug_list = x
        else:
            aug_list = []
        if random.random() > self.p:
            aug_list.append(False)
            return x, aug_list
        aug_list.append(True)
        return self.fn(x), aug_list

class Augmentator(nn.Module):
    def __init__(self, p=0.3):
        super(Augmentator, self).__init__()
        self.p = p
        self.augmentation = nn.Sequential(
            RandomApply(
                x_flip,
                p=self.p
            ),
            RandomApply(
                y_flip,
                p=self.p
            ),
            RandomApply(
                lambda x: gaussian_noise(x, mean=0, std=0.1),
                p=self.p
            ),
            # RandomApply(
            #     lambda x: spec_time(x, max_masked_bins=random.randint(1, 2), mask_size=random.uniform(2, 5)),
            #     p=self.p
            # ),
            # RandomApply(
            #     lambda x: spec_freq(x, max_masked_bins=random.randint(1, 2), mask_size=random.uniform(2, 5)),
            #     p=self.p
            # ),
        )

    def forward(self, x, aug_list):
        curr_augs = aug_list        
        while curr_augs == aug_list:    # Current augmentation must be differ from previous augmentation
            augmented_x, curr_augs = self.augmentation(x)
        return augmented_x, curr_augs
    
    
if __name__ == '__main__':
    x = torch.rand(3, 12, 2500).to('cuda')
    augmentator = Augmentator()
    for i in range(10):
        out1, augs1 = augmentator(x, None)
        out2, augs2 = augmentator(x, augs1)
        print(augs1, augs2)
