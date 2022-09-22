import numpy as np
import pickle
import pandas as pd
import torch
from datetime import datetime
from pytimekr import pytimekr

x = np.load('/nas/datahub/mirae/Data/x_10S_total.npy')
y = np.load('/nas/datahub/mirae/Data/y_10S_total.npy')
Time = np.load('/nas/datahub/mirae/Data/time_10S_index.npy')

x = torch.Tensor(x)
y = torch.Tensor(y)
x = torch.unsqueeze(x, 2)
x = x.repeat(1, 1, 24)
y = torch.unsqueeze(y, 2)
y = y.repeat(1, 1, 24)

kr_holidays_2021 = pytimekr.holidays(year=2021)
kr_holidays_2022 = pytimekr.holidays(year=2022)


for i in range(x.shape[0]):
    for k in range(y.shape[1]):
        date = pd.to_datetime(Time[i][k+x.shape[1]])
        time = date.hour
        month = date.month
        if time in [0, 1, 2]:
            y[i][k][1] = 1
            y[i][k][2] = 0
            y[i][k][3] = 0
            y[i][k][4] = 0
            y[i][k][5] = 0
            y[i][k][6] = 0
            y[i][k][7] = 0
            y[i][k][8] = 0
            y[i][k][9] = 0
            y[i][k][10] = 0
        elif time in [3, 4, 5]:
            y[i][k][1] = 0
            y[i][k][2] = 1
            y[i][k][3] = 0
            y[i][k][4] = 0
            y[i][k][5] = 0
            y[i][k][6] = 0
            y[i][k][7] = 0
            y[i][k][8] = 0
            y[i][k][9] = 0
            y[i][k][10] = 0
        elif time in [6, 7, 8]:
            y[i][k][1] = 0
            y[i][k][2] = 0
            y[i][k][3] = 1
            y[i][k][4] = 0
            y[i][k][5] = 0
            y[i][k][6] = 0
            y[i][k][7] = 0
            y[i][k][8] = 0
            y[i][k][9] = 0
            y[i][k][10] = 0
        elif time in [9, 10, 11]:
            y[i][k][1] = 0
            y[i][k][2] = 0
            y[i][k][3] = 0
            y[i][k][4] = 1
            y[i][k][5] = 0
            y[i][k][6] = 0
            y[i][k][7] = 0
            y[i][k][8] = 0
            y[i][k][9] = 0
            y[i][k][10] = 0
        elif time in [12]:
            y[i][k][1] = 0
            y[i][k][2] = 0
            y[i][k][3] = 0
            y[i][k][4] = 0
            y[i][k][5] = 1
            y[i][k][6] = 0
            y[i][k][7] = 0
            y[i][k][8] = 0
            y[i][k][9] = 0
            y[i][k][10] = 0
        elif time in [13, 14]:
            y[i][k][1] = 0
            y[i][k][2] = 0
            y[i][k][3] = 0
            y[i][k][4] = 0
            y[i][k][5] = 0
            y[i][k][6] = 1
            y[i][k][7] = 0
            y[i][k][8] = 0
            y[i][k][9] = 0
            y[i][k][10] = 0
        elif time in [15, 16, 17]:
            y[i][k][1] = 0
            y[i][k][2] = 0
            y[i][k][3] = 0
            y[i][k][4] = 0
            y[i][k][5] = 0
            y[i][k][6] = 0
            y[i][k][7] = 1
            y[i][k][8] = 0
            y[i][k][9] = 0
            y[i][k][10] = 0
        elif time in [18]:
            y[i][k][1] = 0
            y[i][k][2] = 0
            y[i][k][3] = 0
            y[i][k][4] = 0
            y[i][k][5] = 0
            y[i][k][6] = 0
            y[i][k][7] = 0
            y[i][k][8] = 1
            y[i][k][9] = 0
            y[i][k][10] = 0
        elif time in [19, 20]:
            y[i][k][1] = 0
            y[i][k][2] = 0
            y[i][k][3] = 0
            y[i][k][4] = 0
            y[i][k][5] = 0
            y[i][k][6] = 0
            y[i][k][7] = 0
            y[i][k][8] = 0
            y[i][k][9] = 1
            y[i][k][10] = 0
        elif time in [21, 22, 23]:
            y[i][k][1] = 0
            y[i][k][2] = 0
            y[i][k][3] = 0
            y[i][k][4] = 0
            y[i][k][5] = 0
            y[i][k][6] = 0
            y[i][k][7] = 0
            y[i][k][8] = 0
            y[i][k][9] = 0
            y[i][k][10] = 1
        
        YMD = datetime.strptime(str(date)[:10], '%Y-%m-%d')
        day = YMD.weekday()

        if day == 0:
            y[i][k][11] = 1
            y[i][k][12] = 0
            y[i][k][13] = 0
            y[i][k][14] = 0
            y[i][k][15] = 0
            y[i][k][16] = 0
            y[i][k][17] = 0
        elif day == 1:
            y[i][k][11] = 0
            y[i][k][12] = 1
            y[i][k][13] = 0
            y[i][k][14] = 0
            y[i][k][15] = 0
            y[i][k][16] = 0
            y[i][k][17] = 0
        elif day == 2:
            y[i][k][11] = 0
            y[i][k][12] = 0
            y[i][k][13] = 1
            y[i][k][14] = 0
            y[i][k][15] = 0
            y[i][k][16] = 0
            y[i][k][17] = 0
        elif day == 3:
            y[i][k][11] = 0
            y[i][k][12] = 0
            y[i][k][13] = 0
            y[i][k][14] = 1
            y[i][k][15] = 0
            y[i][k][16] = 0
            y[i][k][17] = 0
        elif day == 4:
            y[i][k][11] = 0
            y[i][k][12] = 0
            y[i][k][13] = 0
            y[i][k][14] = 0
            y[i][k][15] = 1
            y[i][k][16] = 0
            y[i][k][17] = 0
        elif day == 5:
            y[i][k][11] = 0
            y[i][k][12] = 0
            y[i][k][13] = 0
            y[i][k][14] = 0
            y[i][k][15] = 0
            y[i][k][16] = 1
            y[i][k][17] = 0
        elif day == 6:
            y[i][k][11] = 0
            y[i][k][12] = 0
            y[i][k][13] = 0
            y[i][k][14] = 0
            y[i][k][15] = 0
            y[i][k][16] = 0
            y[i][k][17] = 1

        if month in [11, 12, 1, 2]:
            y[i][k][18] = 1
            y[i][k][19] = 0
            y[i][k][20] = 0
            y[i][k][21] = 0
        elif month in [3, 4, 5]:
            y[i][k][18] = 0
            y[i][k][19] = 1
            y[i][k][20] = 0
            y[i][k][21] = 0
        elif month in [6, 7, 8, 9]:
            y[i][k][18] = 0
            y[i][k][19] = 0
            y[i][k][20] = 1
            y[i][k][21] = 0
        elif month in [10]:
            y[i][k][18] = 0
            y[i][k][19] = 0
            y[i][k][20] = 0
            y[i][k][21] = 1
    
        
        if YMD in kr_holidays_2021:
            y[i][k][22] = 1
            y[i][k][23] = 0
        elif YMD in kr_holidays_2022:
            y[i][k][22] = 1
            y[i][k][23] = 0
        else:
            y[i][k][22] = 0
            y[i][k][23] = 1
            
    print(i)

np.save('../Data/y_one_hot_10S.npy', y)

print(y.shape)
print(y[0])
