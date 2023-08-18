import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader

import utils

# 시계열 데이터셋 클래스 정의
class TimeSeriesDataset(torch.utils.data.Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        if self.transform:
            sample = self.transform(sample)
        return sample

# BYOL Loss 정의
class BYOLLoss(nn.Module):
    def __init__(self, temperature=0.5):
        super(BYOLLoss, self).__init__()
        self.temperature = temperature

    def forward(self, z_i, z_j):
        z_i_norm = nn.functional.normalize(z_i, dim=1, p=2)
        z_j_norm = nn.functional.normalize(z_j, dim=1, p=2)
        loss = 2 - 2 * (z_i_norm * z_j_norm).sum(dim=-1)
        return loss.mean()


args = utils.parse_args()

device     = 'cuda' if torch.cuda.is_available() else 'cpu'

# 가상의 시계열 데이터 생성
dataset    = utils.load_dataset(args, is_train=True)
dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

# 모델 및 손실 함수 초기화
target_model = utils.build_model(args)
online_model = utils.build_model(args)
target_model.to(device)
online_model.to(device)

temperature = 0.5
byol_loss = BYOLLoss(temperature).to(device)
optimizer = utils.build_optimizer(online_model, args)

# 학습 루프
for epoch in tqdm(range(args.epochs)):
    online_model.train()
    target_model.eval()
    total_loss = 0.0

    for X, _ in dataloader:
        X = X.to(device)
        optimizer.zero_grad()

        z_i = online_model(X)
        with torch.no_grad():
            z_j = target_model(X)
        
        loss = byol_loss(z_i, z_j)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()

    print(f"Epoch [{epoch+1}/{args.epochs}], Loss: {total_loss / len(dataloader):.4f}")
