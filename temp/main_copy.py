
import os
import tqdm
import torch
import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import dates
import pickle
import pandas as pd
import os 

from earlystopping import EarlyStopping
from dataset import MiraeDataset
from model import lstm_encoder_decoder, lstm_ode_model
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

# seed 고정
def fix_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.use_deterministic_algorithms(True)
    os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:8"
fix_seed(0)

# data load
x = np.load('../Data/x_one_hot_10S.npy')
y = np.load('../Data/y_one_hot_10S.npy')
time = np.load('/nas/datahub/mirae/Data/time_10S_index.npy')
# x = np.load('/nas/datahub/mirae/Data/x_total.npy')
# x = np.expand_dims(x, axis=-1)
# y = np.load('/nas/datahub/mirae/Data/y_total.npy')
# y = np.expand_dims(y, axis=-1)

# data split
# train_idx, test_idx, _, _ = train_test_split(list(range(len(x))), y, test_size=.1, random_state=42, shuffle=True)
# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.1, random_state=42, shuffle=True)
# train_idx, time_idx, _, _ = train_test_split(list(range(len(x_train))), y[:5852], test_size=.1, random_state=42, shuffle=True)
# x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=.1, random_state=42, shuffle=True)

train_len = int(x.shape[0] * 0.7)
val_len = int(x.shape[0] * 0.85)
x_train, y_train = x[:train_len], y[:train_len]
x_val, y_val = x[train_len:val_len], y[train_len:val_len]
x_test, y_test = x[val_len:], y[val_len:]
test_time = time[val_len:]


# dataloader
train_dataset = MiraeDataset(x_train, y_train)
val_dataset = MiraeDataset(x_val, y_val, train_dataset.max_x, train_dataset.min_x, is_train=False)
test_dataset = MiraeDataset(x_test, y_test, train_dataset.max_x, train_dataset.min_x, is_train=False)

train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=2)

# model
model = lstm_encoder_decoder(24, 128).cuda()
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
        outputs = model(inputs, labels, 3, 0.5)
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

def evaluate(model, loader):
    model.eval()
    epoch_nor_loss = 0
    epoch_denor_loss = 0
    total_nor_loss = []
    total_denor_loss = []
    output_nor_list = []
    output_denor_list = []
    with torch.no_grad():
        for idx, data in enumerate(tqdm.tqdm(loader, desc=f'{1} epoch')):
            inputs, labels = data
            inputs, labels = inputs.cuda(), labels.cuda()
            outputs = model.predict(inputs, 3).cuda()
            output_nor_list.append(outputs)
            nor_loss = criterion(outputs, labels)
            denor_labels = labels * (train_dataset.max_x - train_dataset.min_x) + train_dataset.min_x
            denor_outputs = outputs * (train_dataset.max_x - train_dataset.min_x) + train_dataset.min_x
            output_denor_list.append(denor_outputs)
            denor_loss = criterion(denor_outputs, denor_labels)
            total_nor_loss.append(nor_loss)
            total_denor_loss.append(denor_loss)
            epoch_nor_loss += nor_loss.mean().item()
            epoch_denor_loss += denor_loss.mean().item()
        eval_nor_loss = epoch_nor_loss/len(loader)
        eval_denor_loss = epoch_denor_loss/len(loader)
    return total_nor_loss, total_denor_loss, output_nor_list, output_denor_list, eval_nor_loss, eval_denor_loss

# Train
# early_stopping = EarlyStopping(patience=200)
# for epoch in range(20):
#     train_nor_loss, train_denor_loss, train_loss = train(model, train_loader, optimizer, criterion)
#     total_nor_loss, total_denor_loss, output_nor_list, total_denor_list, eval_nor_loss, eval_denor_loss = evaluate(model, val_loader)

#     # EarlyStopping에서 loss 확인
#     early_stopping.step(eval_nor_loss)
#     # 만약 loss가 5번동안 비슷하면 Break
#     if early_stopping.is_stop():
#         print(f'Training Loss of Normalized Data: {train_nor_loss}')
#         print(f'Training Loss of Denormalized Data: {train_denor_loss}')
#         print(f'Validation Loss of Normalized Data: {eval_nor_loss}')
#         print(f'Validation Loss of Denormalized Data: {eval_denor_loss}')
#         print(f'Current Epoch: {epoch}')
#         break
#     else:
#         print(f'Training Loss of Normalized Data: {train_nor_loss}')
#         print(f'Training Loss of Denormalized Data: {train_denor_loss}')
#         print(f'Validation Loss of Normalized Data: {eval_nor_loss}')
#         print(f'Validation Loss of Denormalized Data: {eval_denor_loss}')
# torch.save(model, './models/Seq2Seq_10S_24dim.pt')

# test
model = torch.load('./models/Seq2Seq_10S_24dim.pt')
total_nor_loss, total_denor_loss, output_nor_list, total_denor_list, eval_nor_loss, eval_denor_loss = evaluate(model, test_loader)
print(f'Test Loss of Normalized Data: ', eval_nor_loss)
print(f'Test Loss of Denormalized Data: ', eval_denor_loss)

total_loss = torch.cat(total_nor_loss, dim=0).sum(dim=1)
top_5_ind = torch.topk(total_loss, dim=0, k=5)[1]
bottom_5_ind = torch.topk(-total_loss, dim=0, k=5)[1]
output_tensor = torch.cat(output_nor_list, dim=0)

total_dic = dict()
for i in range(len(test_time)):
    date = pd.to_datetime(test_time[i][0])
    if (date.hour in [11, 17]) and (49 == date.minute):
        total_dic[i] = total_loss[i][0].item()
final_loss_top = sorted(total_dic.items(), reverse=True)[:50]
final_loss_bottom = sorted(total_dic.items())[:50]

# # 큰 loss 뽑기
# model = torch.load('Seq2Seq_dim.pt')
# criterion = torch.nn.MSELoss(reduction='none').cuda()
# optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
# test_loss, _, _, _, _, _ = evaluate(model, test_loader, optimizer)
# val_loss, _, _, _, _, _ = evaluate(model, val_loader, optimizer)
# _, _, train_loss = train(model, train_loader, optimizer, criterion)

# test_loss = torch.cat(test_loss, dim=0)
# val_loss = torch.cat(val_loss, dim=0)
# total_loss = torch.cat([val_loss, test_loss], dim=0).sum(dim=1)

# # 시간 index
# time_idx.extend(test_idx)

# # top 200
# top_200_idx = torch.topk(total_loss, dim=0, k=200)[1]

# frq_month = []
# for i in range(200):
#     idx = top_200_idx[i][0]
#     month = pd.to_datetime(time[time_idx[i]]).month
#     frq_month.append(month[0])

# frq_x = [i for i in range(1, 13)]
# frq_y = [frq_month.count(i) for i in range(1, 13)]
# plt.title('Inaccurate Months of the Seq2Seq')
# plt.bar(frq_x, frq_y)
# plt.xticks(frq_x)
# for i in range(1, len(frq_y)+1):
#     plt.text(i, frq_y[i-1]+0.25, frq_y[i-1], ha='center', size=11)
# plt.xlabel('Month')
# plt.ylabel('Count')
# plt.savefig(f'./result_22/result_hist.png')

# visualization
# denormalize
inputs = x_test
labels = y_test
outputs = output_tensor * (train_dataset.max_x - train_dataset.min_x) + train_dataset.min_x

# # Visualization: Top5 (min)
# plt.rcParams["figure.figsize"] = (16,6)
# for i in range(5):
#     idx = bottom_5_ind[i][0]
#     inputs_idx = inputs[idx][:,0]
#     labels_idx = labels[idx][:,0]
#     outputs_idx = outputs[idx][:,0].cpu()
#     input_time = time[idx][0:60]
#     label_time = time[idx][60:63]
#     plt.clf()
#     plt.plot(input_time, inputs_idx, 'b', label='x')
#     plt.plot(label_time, labels_idx, 'bo', ms=5, alpha=0.7, label='y')
#     plt.plot(label_time, outputs_idx, 'ro', ms=5, alpha=0.7, label='y_hat')
#     plt.title(f'Result of Test data Top {i+1} (Min): {total_loss[idx][0]/3:.5f}', fontsize=25)
#     plt.legend()
#     plt.savefig(f'./result_10S_24/result_min_{i+1}_{idx}.png')

# Visualization: Top5 (max)
plt.rcParams["figure.figsize"] = (20,12)
for i in range(5):
    idx = final_loss_top[i+10][0]
    inputs_idx = inputs[idx][:,0]
    labels_idx = labels[idx][:,0]
    outputs_idx = outputs[idx][:,0].cpu()
    input_time = test_time[idx][0:60]
    label_time = test_time[idx][60:63]
    plt.clf()
    plt.plot(input_time, inputs_idx,'b', label='x')
    plt.plot(label_time, labels_idx, 'bo', ms=5, alpha=0.7, label='y')
    plt.plot(label_time, outputs_idx, 'ro', ms=5, alpha=0.7, label='y_hat')
    hfmt = dates.DateFormatter('%m/%d %H:%M:%S')
    ax = plt.gca()
    # fig, ax = plt.subplots()
    # ax.plot(input_time, inputs_idx)
    # ax.plot(label_time, labels_idx)
    # ax.plot(label_time, outputs_idx)
    ax.xaxis.set_major_formatter(hfmt)
    ax.xaxis.set_major_locator(dates.MinuteLocator())
    ax.set_title(f'Result of Test data Top {i+1} (Max): {final_loss_top[i][1]/3:.5f}', fontsize=25)
    # plt.legend(loc=2)
    plt.xticks(rotation=45)
    plt.show()
    plt.savefig(f'./result_10S_24/result_max_{i+1}_{idx}.png')

# # normalize for profile
# Ytest = (y_test-train_dataset.min_x)/(train_dataset.max_x - train_dataset.min_x)
# labels = y_test

# # Profile
# week_profile = {0: 42.519685, 1:42.519685, 2:42.519685, 3:42.519685, 4:49.606299, 5:49.606299, 6:49.606299, 7:63.779528, 8:77.952756, 9:99.212598, 10:113.385827, 11:92.125984, 12:113.385827, 13:113.385827, 14:120.472441, 15:127.559055, 16:106.299213, 17:77.952756, 18:99.212598, 19:85.039370, 20:56.692913, 21:49.606299, 22:42.519685, 23:42.519685}
# weekend_profile = {0:50.769231, 1:50.769231, 2:50.769231, 3:50.769231, 4:50.769231, 5:50.769231, 6:59.230769, 7:59.230769, 8:59.230769, 9:59.230769, 10:59.230769, 11:59.230769, 12:59.230769, 13:59.230769, 14:59.230769, 15:59.230769, 16:59.230769, 17:59.230769, 18:50.769231, 19:50.769231, 20:50.769231, 21:50.769231, 22:50.769231, 23:50.769231}

# # Profile Loss
# profile = torch.zeros_like(torch.Tensor(labels))
# for i in range(len(labels)):
#     for j in range(len(labels[i][:,2])):
#         if labels[i][:,2][j] in [0, 1, 2, 3, 4, 5, 6]:
#             profile[i][j][0] = week_profile[labels[i][:,1][j]]
#         else:
#             profile[i][j][0] = weekend_profile[labels[i][:,1][j]]
# np.save('./profile_no_split.npy', profile)

# # Profile
# profile = np.load('./profile_no_split.npy')
# profile_nor = (profile-train_dataset.min_x)/(train_dataset.max_x - train_dataset.min_x)

# # Profile loss 계산
# profile_total_loss = []
# Labels = torch.Tensor(Ytest[:,:,0])
# Profiles = torch.Tensor((profile_nor[:,:,0]))
# loss = torch.nn.MSELoss(reduction='none')
# profile_loss = loss(Labels, Profiles)
# profile_total_loss.append(profile_loss)
# final_loss = profile_loss.mean().item()
# print(f'Profile Loss: {final_loss}')


# # Max Top5
# profile_total_loss = torch.cat(profile_total_loss, dim=0).sum(dim=1)
# top_5_ind = torch.topk(profile_total_loss, k=5)[1]
# bottom_5_ind = torch.topk(-profile_total_loss, k=5)[1]

# # Visualization of Profile: Top5
# plt.rcParams["figure.figsize"] = (16,6)
# for i in range(5):
#     idx = bottom_5_ind[i] # 바꿔줘야 함
#     inputs_idx = inputs[idx][:,0]/720
#     labels_idx = labels[idx][:,0]/720
#     outputs_idx = profile[idx][:,0]
#     input_time = time[idx][0:72]
#     label_time = time[idx][72:84]
#     plt.clf()
#     plt.plot(input_time, inputs_idx, 'b', label='x')
#     plt.plot(label_time, labels_idx, 'bo', ms=5, alpha=0.7, label='y')
#     plt.plot(label_time, outputs_idx, 'ro', ms=5, alpha=0.7, label='y_hat')
#     plt.title(f'Result of Test data Top {i+1} (Min): {profile_total_loss[idx]/12:.5f}', fontsize=25) # 바꿔줘야 함
#     plt.legend()
#     plt.savefig(f'./result_22/result_min_profile_{i+1}_{idx}.png')
