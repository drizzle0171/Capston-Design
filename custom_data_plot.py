import torch
import numpy as np
import os 
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib import gridspec
from matplotlib import dates
import torch
import random
import pandas as pd

# df_electricity = np.array(pd.read_csv('./dataset/electricity.csv'))
# electricity = [random.randint(0, df_electricity.shape[1]) for i in range(3)]
# x_time = list(range(336))

# plt.rcParams["figure.figsize"] = (18,12)
# for i in electricity:
#     plt.clf()
#     # train
#     plt.plot(x_time, df_electricity[:336, i], 'b-', alpha=0.7)

#     plt.title(f'Electricity-{i}', fontsize=20)
#     plt.ylabel('Electricity', fontsize=15)
#     plt.xlabel('Time Steps', fontsize=15)
#     plt.savefig(f'./visual/electricity_{i}.png')
    
df_mirae = np.load('/nas/datahub/mirae/Data/x_H_total.npy')
mirae = [random.randint(0, df_mirae.shape[0]) for i in range(3)]
x_time = list(range(72))

plt.rcParams["figure.figsize"] = (18,12)
for i in mirae:
    plt.clf()
    # train
    plt.plot(x_time, df_mirae[i, :], 'b-', alpha=0.7)
    plt.title(f'mirae-{i}', fontsize=20)
    plt.ylabel('Power', fontsize=15)
    plt.xlabel('Time Steps', fontsize=15)
    plt.savefig(f'./visual/mirae_{i}.png')
    