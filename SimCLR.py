import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm
import torch.optim as optim
from torch.utils.data import DataLoader

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

# 모델 정의
class Encoder(nn.Module):
    def __init__(self, input_channels, sequence_length, embedding_size):
        super(Encoder, self).__init__()
        self.embedding = nn.Sequential(
            nn.Conv1d(input_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(sequence_length * 64, embedding_size)
        )

    def forward(self, x):
        return self.embedding(x)

# SimCLR Loss 정의
class SimCLRLoss(nn.Module):
    def __init__(self, temperature=0.5):
        super(SimCLRLoss, self).__init__()
        self.temperature = temperature

    def forward(self, z_i, z_j):
        z = torch.cat((z_i, z_j), dim=0)
        sim_matrix = torch.mm(z, z.t())
        mask = torch.eye(z.size(0), dtype=bool)
        sim_matrix_masked = sim_matrix[~mask].view(z.size(0), -1)
        logits = torch.exp(sim_matrix_masked / self.temperature)
        loss = -torch.log(logits / logits.sum(1, keepdim=True)).mean()
        return loss


device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 하이퍼파라미터 및 데이터 초기화
input_channels = 12
sequence_length = 2500
embedding_size = 64
temperature = 0.5
batch_size = 32
num_epochs = 10

# 가상의 멀티채널 시계열 데이터 생성
num_samples = 1000
data = torch.randn(num_samples, input_channels, sequence_length)

# 데이터 로더 초기화
train_dataset = TimeSeriesDataset(data)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# 모델 및 손실 함수 초기화
encoder = Encoder(input_channels, sequence_length, embedding_size).to(device)
simclr_loss = SimCLRLoss(temperature).to(device)
optimizer = optim.Adam(encoder.parameters(), lr=0.001)


print("SETTING COMPLETE!")

# 학습 루프
for epoch in range(num_epochs):
    encoder.train()
    total_loss = 0.0

    for batch in train_loader:
        optimizer.zero_grad()

        z_i = encoder(batch)
        
        # 데이터 증강을 통한 positive pair 생성
        perm_positive = np.random.permutation(batch.size(0))
        batch_positive = batch[perm_positive]
        z_j_positive = encoder(batch_positive)
        
        # 데이터 증강을 통한 negative pair 생성 (시간적 순서 유지)
        perm_negative = torch.randperm(batch.size(1))  # 시간 축을 섞음
        batch_negative = batch[:, perm_negative, :]
        z_j_negative = encoder(batch_negative)
        
        loss_positive = simclr_loss(z_i, z_j_positive)
        loss_negative = simclr_loss(z_i, z_j_negative)
        loss = loss_positive + loss_negative
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss / len(train_loader):.4f}")