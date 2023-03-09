import torch
import numpy as np
import os 
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib import gridspec
from matplotlib import dates
import torch

# color map
for i in ['softmax', 'sigmoid', 'softmax']:
    realtime_x = np.load('/nas/datahub/mirae/Data/time_H_index.npy')[:32][13][:72]
    realtime_y = np.load('/nas/datahub/mirae/Data/time_H_index.npy')[:32][13][72:]
    real_power = np.load('/nas/datahub/mirae/Data/x_H_total.npy')[:32][13]
    real_label = np.load('/nas/datahub/mirae/Data/y_H_total.npy')[:32][13]
    time = np.load(f'./time_param/time_{i}.npy')
    time = time[0][13]
    
    save_root = './time_weights_result/'
    if not os.path.exists(save_root):
        os.mkdir(save_root)

    fig,ax=plt.subplots()
    time = time.T
    im=ax.imshow(time,cmap='plasma_r'))
    fig.colorbar(im,pad=0.03)
    plt.title(f'Using {i}')
    plt.xlabel('Seqeunce')
    plt.ylabel('Batch')
    plt.savefig(os.path.join(save_root,i + '_1.jpg'),dpi=500)
    plt.close()

# power with tempoal weight
for i in ['softmax']:
    realtime_x = np.load('/nas/datahub/mirae/Data/time_H_index.npy')[:32][13][:72]
    realtime_y = np.load('/nas/datahub/mirae/Data/time_H_index.npy')[:32][13][72:]
    real_power = np.load('/nas/datahub/mirae/Data/x_H_total.npy')[:32][13]
    real_label = np.load('/nas/datahub/mirae/Data/y_H_total.npy')[:32][13]
    time = np.load(f'./time_param/time_{i}.npy')
    time = time[0][13].squeeze()
    
    save_root = './time_weights_result/'
    if not os.path.exists(save_root):
        os.mkdir(save_root)
        
    # fig, (ax1, ax2) = plt.subplots(2,1,sharex=True)
    # fig.subplots_adjust(hspace=0.1)
    
    fig = plt.figure(figsize=(20, 15)) 
    gs = gridspec.GridSpec(nrows=2, # row 몇 개 
                       ncols=1, # col 몇 개 
                       height_ratios=[8, 2],
                    #    width_ratios=[7.5,7.5]
                      )
    
    ax1 = plt.subplot(gs[0])
    ax1.plot(realtime_x, real_power, lw=5)
    ax1.plot(realtime_y, real_label, lw=5)
    
    ax2 = plt.subplot(gs[1])
    ax2.bar(realtime_x, time, width=0.1)
    
    # ax1.set_yticks([50, 200], fontsize=25)
    # ax2.set_xticks([0,0.2], fontsize=25)
    ax1.set_ylim(50, 200)
    ax2.set_ylim(0,0.2)
    ax1.tick_params(axis='y', labelsize=20)
    ax2.tick_params(axis='y', labelsize=20)

    
        # 두 그래프 사이의 경계선 제거
    ax1.spines['bottom'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax1.xaxis.tick_top()
    ax1.tick_params(labeltop=False)
    ax2.xaxis.tick_bottom()

    # 두 그래프 사이의 y축에 물결선 효과 마커 표시
    kwargs = dict(marker=[(-1, -0.5), (1, 0.5)], markersize=12,
                linestyle="none", color='k', mec='k', mew=1, clip_on=False)
    ax1.plot([0, 1], [0, 0], transform=ax1.transAxes, **kwargs)
    ax2.plot([0, 1], [1, 1], transform=ax2.transAxes, **kwargs)
    ax = plt.gca()
    ax.xaxis.set_major_locator(dates.HourLocator(interval=3))  
    ax1.set_title("The power & temporal weight at the moment",fontsize=35, pad=20)     
    plt.show()
    plt.xticks(fontsize=20,rotation=45)
    plt.savefig(os.path.join(save_root,i + '_1.jpg'),dpi=500)