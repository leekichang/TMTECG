"""
Created on Thu Aug 18 2023
@author: Kichang Lee
@contact: kichan.lee@yonsei.ac.kr
"""

import torch
import random
import torch.nn as nn

__all__ = [
    'Augmentator'
]

def x_flip(
    x:torch.FloatTensor
)->torch.FloatTensor:
    """
    x: (N,C,S)
    return: (N,C,S)
    
    Vertical (x-axis) Flip
    """
    return x*-1

def y_flip(
    x:torch.FloatTensor
)->torch.FloatTensor:
    """
    x: (N,C,S)
    return: (N,C,S)
    
    Horizontal (y-axis) Flip
    """
    return x.flip(dims=[-1])

def gaussian_noise(
    x:torch.FloatTensor,
    mean:float=0,
    std:float=0.1
)->torch.FloatTensor:
    """
    x: (N,C,S)
    return: (N,C,S)
    
    Add Guassian Noise
    """
    noise = (torch.randn(x.shape) * std + mean).to('cuda' if torch.cuda.is_available() else 'cpu')
    return x + noise

class RandomApply(nn.Module):
    def __init__(self, fn, p):
        super().__init__()
        self.fn = fn
        self.p = p
    def forward(self, x):
        if random.random() > self.p:
            return x
        return self.fn(x)

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
        )

    def forward(self, x):
        augmented_x = self.augmentation(x)
        return augmented_x
    
    
if __name__ == '__main__':
    pass