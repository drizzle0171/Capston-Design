
import os
import tqdm
import torch
import random
import numpy as np
import matplotlib.pyplot as plt
import os 

from dataset import MiraeDataset
from model import lstm_encoder_decoder
from torch.utils.data import DataLoader

# seed 고정
def fix_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.use_deterministic_algorithms(True)
    os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:8"
fix_seed(0)

# data load
x = np.load('../Data/x_one_hot_H.npy')
y = np.load('../Data/y_one_hot_H.npy')
time = np.load('/nas/datahub/mirae/Data/time_index.npy')

# dataloader
train_dataset = MiraeDataset(x, y)
train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=2)

# model
model = lstm_encoder_decoder(22, 128).cuda()
criterion = torch.nn.MSELoss(reduction='none').cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# train 함수 정의
def train(model, loader, optimizer, criterion):
    model.train()
    train_loss = []
    epoch_nor_loss = 0
    epoch_denor_loss = 0
    for idx, data in enumerate(tqdm.tqdm(loader, desc=f'{epoch+1} epoch')):
        inputs, labels = data
        inputs, labels = inputs.cuda(), labels.cuda()
        outputs = model(inputs, labels, 12, 0.5)
        optimizer.zero_grad()
        nor_loss = criterion(outputs, labels).mean()
        train_loss.append(nor_loss)
        nor_loss.backward()
        denor_labels = labels * (train_dataset.max_x - train_dataset.min_x) + train_dataset.min_x
        denor_outputs = outputs * (train_dataset.max_x - train_dataset.min_x) + train_dataset.min_x
        denor_loss = criterion(denor_outputs, denor_labels)
        optimizer.step()
        epoch_nor_loss += nor_loss.mean().item()
        epoch_denor_loss += denor_loss.mean().item()
    train_nor_loss = epoch_nor_loss/len(loader)
    train_denor_loss = epoch_denor_loss/len(loader)
    return train_nor_loss, train_denor_loss, train_loss

# Train
for epoch in range(200):
    train_nor_loss, train_denor_loss, train_loss = train(model, train_loader, optimizer, criterion)
    print(f'Training Loss of Normalized Data: {train_nor_loss}')
    print(f'Training Loss of Denormalized Data: {train_denor_loss}')
torch.save(model, './models/Seq2Seq_final.pt')
