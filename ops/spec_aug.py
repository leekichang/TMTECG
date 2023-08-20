import torch
import numpy as np

def random_time_masking(stft_data, max_masked_bins=10):
    masked_stft_data = stft_data.clone()
    num_samples, num_channels, freq_bins, time_bins = stft_data.shape

    for i in range(num_samples):
        for j in range(num_channels):
            num_masked_bins = np.random.randint(0, max_masked_bins + 1)
            masked_bins = np.random.choice(time_bins, num_masked_bins, replace=False)
            masked_stft_data[i, j, :, masked_bins] = 0.0

    return masked_stft_data

def random_frequency_masking(stft_data, max_masked_bins=10):
    masked_stft_data = stft_data.clone()
    num_samples, num_channels, freq_bins, time_bins = stft_data.shape

    for i in range(num_samples):
        for j in range(num_channels):
            num_masked_bins = np.random.randint(0, max_masked_bins + 1)
            masked_bins = np.random.choice(freq_bins, num_masked_bins, replace=False)
            masked_stft_data[i, j, masked_bins, :] = 0.0

    return masked_stft_data