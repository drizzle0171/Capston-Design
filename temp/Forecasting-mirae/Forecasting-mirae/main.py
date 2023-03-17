
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
from preparing import preparing

# seed 고정
def fix_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.use_deterministic_algorithms(True)
    os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:8"
fix_seed(0)

# data load - test
######### Inference할 데이터 입력 #######
x = np.load('')
y = np.load('')
time = np.loae('')
#####################################

# Feature 추가
preparing(x, y, time)
x_test = np.load('./test_data/x.npy')
y_test = np.load('./test_data/y.npy')

# data load - train: 데이터 정규화에 사용
x_train = np.load('./train_data/x_train.npy')
y_train = np.load('./train_data/y_train.npy')

# dataloader
train_dataset = MiraeDataset(x_train, y_train)
test_dataset= MiraeDataset(x_test, y_test, train_dataset.max_x, train_dataset.min_x, is_train=False)
test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=2)

# model
model = torch.load('Seq2Seq_final.pt')

# eveluate 함수 정의
def evaluate(model, loader):
    model.eval()
    output_denor_list = []
    with torch.no_grad():
        for idx, data in enumerate(tqdm.tqdm(loader, desc=f'{1} epoch')):
            inputs, labels = data
            inputs, labels = inputs.cuda(), labels.cuda()
            outputs = model.predict(inputs, 12).cuda()
            denor_outputs = (outputs * (train_dataset.max_x - train_dataset.min_x) + train_dataset.min_x)
            output_denor_list.append(denor_outputs)
    return output_denor_list

# test: output 도출
output_list = torch.tensor(evaluate(model, test_loader))
print(output_list)