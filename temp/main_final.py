
import os
import tqdm
import torch
import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import dates
import pandas as pd
import os 
import tqdm

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

# data load: hour
xh = np.load('../Data/x_H.npy')
yh = np.load('../Data/y_H.npy')
timeh = np.load('/nas/datahub/mirae/Data/time_H_index.npy')
timeH = timeh[:,0]

# data load: 10min
xm = np.load('../Data/x_H-10M.npy')
ym = np.load('../Data/y_H-10M.npy')
timem = np.load('/nas/datahub/mirae/Data/time-H_10Min_index.npy')
timeM = timem[:,0]

# data load: 10sec
xs = np.load('../Data/x_H-10M-10s.npy')
ys = np.load('../Data/y_H-10M-10s.npy')
times = np.load('/nas/datahub/mirae/Data/time-H-10Min_10s_index.npy')
timeS = times[:,0]

print("Data are ready")

# data split: hour
train_len = int(xh.shape[0] * 0.7)
val_len = int(xh.shape[0] * 0.85)
x_train, y_train = xh[:train_len], yh[:train_len]
x_test, y_test = xh[val_len:], yh[val_len:]
test_time_h = timeh[val_len:]
train_h = MiraeDataset(x_train, y_train)
test_h = MiraeDataset(x_test, y_test, train_h.max_x, train_h.min_x, is_train=False)
test_h_loader = DataLoader(test_h, batch_size=256, shuffle=False, num_workers=2)


# data split: 10min
train_len = int(xm.shape[0] * 0.7)
val_len = int(xm.shape[0] * 0.85)
x_train, y_train = xm[:train_len], ym[:train_len]
x_test, y_test = xm[val_len:], ym[val_len:]
test_time_m = timem[val_len:]
train_m = MiraeDataset(x_train, y_train)
test_m = MiraeDataset(x_test, y_test, train_m.max_x, train_m.min_x, is_train=False)
test_m_loader = DataLoader(test_m, batch_size=256, shuffle=False, num_workers=2)


# data split: hour
train_len = int(xs.shape[0] * 0.7)
val_len = int(xs.shape[0] * 0.85)
x_train, y_train = xs[:train_len], ys[:train_len]
x_test, y_test = xs[val_len:], ys[val_len:]
test_time_s = times[val_len:]
train_s = MiraeDataset(x_train, y_train)
test_s = MiraeDataset(x_test, y_test, train_s.max_x, train_s.min_x, is_train=False)
test_s_loader = DataLoader(test_s, batch_size=256, shuffle=False, num_workers=2)


# model
model = lstm_encoder_decoder(22, 128).cuda()
criterion = torch.nn.MSELoss(reduction='none').cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
print("model and etc are ready")

# Output을 위한 예측 함수
def evaluate(model, loader):
    model.eval()
    epoch_nor_loss = 0
    total_nor_loss = []
    output_nor_list = []
    with torch.no_grad():
        for idx, data in enumerate(tqdm.tqdm(loader, desc=f'{1} epoch')):
            # 데이터 불러오기
            inputs, labels = data
            inputs, labels = inputs.cuda(), labels.cuda()

            # normalized output & loss
            outputs = model.predict(inputs, 12).cuda()
            output_nor_list.append(outputs)
            nor_loss = criterion(outputs, labels)

            # loss 저장
            total_nor_loss.append(nor_loss)

            # epoch loss 합 
            epoch_nor_loss += nor_loss.mean().item()
        
        # epoch loss 평균
        eval_nor_loss = epoch_nor_loss/len(loader)

    return total_nor_loss, output_nor_list, eval_nor_loss

def final():
    model_h = torch.load('./models/Seq2Seq_H.pt')
    model_m = torch.load('./models/Seq2Seq_10M.pt')
    model_s = torch.load('./models/Seq2Seq_10S_eph20.pt')
    _, output_nor_list_H, _ = evaluate(model_h, test_h_loader)
    _, output_nor_list_10M, _ = evaluate(model_m, test_m_loader)
    _, output_nor_list_10S, _ = evaluate(model_s, test_s_loader)

    output_H = (torch.cat(output_nor_list_H, dim=0) * (train_h.max_x - train_h.min_x) + train_h.min_x)[:,:,0]
    output_10M = (torch.cat(output_nor_list_10M, dim=0) * (train_m.max_x - train_m.min_x) + train_m.min_x)[:,:,0]
    output_10S = (torch.cat(output_nor_list_10S, dim=0) * (train_s.max_x - train_s.min_x) + train_s.min_x)[:,:,0]

    for i in tqdm.tqdm(range(len(output_10S))):
        for j in range(len(output_10S[i])):
            try:
                date = pd.to_datetime(test_time_s[i][j])
                if (date.hour == 23):
                    if (date.day == 30 or date.day == 31):
                        hour = pd.to_datetime(f'{date.year}-{date.month+1}-01 00:00:00')
                    else:
                        hour = pd.to_datetime(f'{date.year}-{date.month}-{date.day+1} 00:00:00')
                else:
                    hour = pd.to_datetime(f'{date.year}-{date.month}-{date.day} {date.hour+1}:00:00')
                idx_h_axis0 = np.where(test_time_h[:,72:]==hour)[0][0]
                idx_h_axis1 = np.where(test_time_h[:,72:]==hour)[1][0]
                output_10S[i][j] = output_10S[i][j].item() + output_H[idx_h_axis0][idx_h_axis1].item()

                if date.minute < 50: 
                    minute = pd.to_datetime(f'{date.year}-{date.month}-{date.day} {date.hour}:{(date.minute//10)*10+10}:00')
                elif (date.hour < 23) and (date.minute) > 50:
                    minute = pd.to_datetime(f'{date.year}-{date.month}-{date.day} {date.hour+1}:00:00')
                elif (date.hour == 23) and (date.minute > 50):
                    if (date.day == 30 or date.day == 31):
                        minute = pd.to_datetime(f'{date.year}-{date.month+1}-01 00:00:00')
                    else:    
                        minute = pd.to_datetime(f'{date.year}-{date.month}-{date.day+1} 00:00:00')
                idx_m_axis0 = np.where(test_time_m[:,72:]==minute)[0][0]
                idx_m_axis1 = np.where(test_time_m[:,72:]==minute)[1][0]
                output_10S[i][j] = output_10S[i][j].item() + output_10M[idx_m_axis0][idx_m_axis1].item()
                import pdb;pdb.set_trace()
                

            except IndexError:
                output_10S[i][j] = float('nan')
                print(date)


    output_10S = output_10S.cpu().numpy()
    np.save('./final_output.npy', output_10S)

    return output_10S

# test
print(final())

# final loss
label = np.load('/nas/datahub/mirae/Data/y_10S_total.npy')
x_10s = np.load('/nas/datahub/mirae/Data/x_10S_total.npy')
val_len = int(x_10s.shape[0] * 0.85)
output = np.load('./final_output.npy')
label = label[val_len:]

label = torch.Tensor(label)
output = torch.Tensor(output)
loss = criterion(output, label)

final_loss = []
for i in range(len(loss)):
    if sum(np.isnan(loss[i])) > 0:
        continue
    else:
        final.append(loss[i])

final_loss = torch.cat(final_loss, dim=0)
print("Final Loss of RMRT: ", final_loss)