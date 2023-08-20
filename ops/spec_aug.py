import torch
import numpy as np

def random_time_masking(stft_data, max_masked_bins=1, mask_size=5):
    masked_stft_data = stft_data.clone()
    num_samples, num_channels, freq_bins, time_bins = stft_data.shape

    for i in range(num_samples):
        for j in range(num_channels):
            num_masked_bins = np.random.randint(1, max_masked_bins + 1)
            masked_bins = np.random.choice(time_bins, num_masked_bins, replace=False)
            for bin_idx in masked_bins:
                bin_range = range(max(0, bin_idx - mask_size), min(time_bins, bin_idx + mask_size + 1))
                masked_stft_data[i, j, :, bin_range] = 0.0

    return masked_stft_data

def random_frequency_masking(stft_data, max_masked_bins=1, mask_size=5):
    masked_stft_data = stft_data.clone()
    num_samples, num_channels, freq_bins, time_bins = stft_data.shape

    for i in range(num_samples):
        for j in range(num_channels):
            num_masked_bins = np.random.randint(1, max_masked_bins + 1)
            masked_bins = np.random.choice(freq_bins, num_masked_bins, replace=False)
            for bin_idx in masked_bins:
                bin_range = range(max(0, bin_idx - mask_size), min(freq_bins, bin_idx + mask_size + 1))
                masked_stft_data[i, j, bin_range, :] = 0.0

    return masked_stft_data