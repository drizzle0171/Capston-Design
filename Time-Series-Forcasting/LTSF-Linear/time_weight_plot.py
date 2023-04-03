import torch
import numpy as np
import os 
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib import gridspec
from matplotlib import dates
import torch
import random


def before_after(function):
    before = np.load(f'./time_param/before_{function}.npy')
    after = np.load(f'./time_param/after_{function}.npy')
    plt.bar(list(range(72)), before.squeeze()[0], label=f'Before {function}')
    plt.bar(list(range(72)), after.squeeze()[0], alpha=0.5, label=f'After {function}')
    plt.legend()
    plt.title(f'Power before and after {function}')
    plt.xlabel('Sequence')
    plt.ylabel('Power')
    plt.savefig(f'./weights_plot/befor_{function}.png',dpi=500)

# color map
def color_map():
    # for i in ['softmax', 'sigmoid', 'linear']:
    origin = np.load(f'./time_param/time_feature.npy')
    for j in [random.randint(0, origin.shape[0]) for k in range(6)]:
        time = origin[j]
        
        save_root = './time_weights_result/'
        if not os.path.exists(save_root):
            os.mkdir(save_root)

        fig,ax=plt.subplots()
        # time = time.T
        im=ax.imshow(time,cmap='plasma_r')
        fig.colorbar(im,pad=0.03)
        plt.title('Time Weight')
        plt.xlabel('Seqeunce')
        plt.ylabel('Batch')
        plt.savefig(os.path.join(save_root,'time_feature' + f'_{j}.jpg'),dpi=500)
        plt.close()

# power with tempoal weight
def power_temporal_weight():
    inputs = np.load('./time_param/input.npy').reshape(-1, 32, 72, 1)
    trues = np.load('./time_param/true.npy').reshape(-1, 32, 12, 1)
    pred = np.load('./time_param/pred.npy').reshape(-1, 32, 12, 1)
    times = np.load('/nas/home/drizzle0171/Time-Series-Forcasting/time_final.npy')
    test_len = int(len(times)*0.85)
    times = times[test_len:][:1216].reshape(-1, 32, 84) # drop last
    for i in ['feature']:
        for j in [4, 9, 29, 31, 33, 37]:
            for k in range(32):
                origin = np.load(f'./time_param/time_feature.npy')
                time = origin[j][k].squeeze()
                y_time = [0]*12
                realtime_x = times[j][k][:72]
                realtime_y = times[j][k][72:]
                real_power = inputs[j][k].squeeze() * (349.8758134722222 - 30.71108222222222) + 30.71108222222222
                real_label = trues[j][k].squeeze() * (349.8758134722222 - 30.71108222222222) + 30.71108222222222
                real_pred = pred[j][k].squeeze() * (349.8758134722222 - 30.71108222222222) + 30.71108222222222

                save_root = './time_weights_result/'
                if not os.path.exists(save_root):
                    os.mkdir(save_root)


                fig = plt.figure(figsize=(20, 15)) 
                gs = gridspec.GridSpec(nrows=2, # row 몇 개 
                                ncols=1, # col 몇 개 
                                height_ratios=[8, 2],
                                #    width_ratios=[7.5,7.5]
                                )
                ax1 = plt.subplot(gs[0])
                ax1.plot(realtime_x, real_power, lw=5, color='b', label='Input power')
                ax1.plot(realtime_y, real_label, lw=0, color='b', marker='o', label='Label power', alpha=0.7, markersize=10)
                ax1.plot(realtime_y, real_pred, lw=0, color='r', marker='o', label='Predicted power', alpha=0.7, markersize=10)
                
                ax2 = plt.subplot(gs[1])
                ax2.bar(realtime_x, time, width=0.1)
                ax2.bar(realtime_y, y_time)
                
                ax1.legend(fontsize=17)
                # ax1.set_yticks([50, 200], fontsize=25)
                # ax2.set_xticks([0,0.2], fontsize=25)
                # ax1.set_ylim(50, 200)
                # ax2.set_ylim(0,0.2)
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
                plt.savefig(os.path.join(save_root, 'final' + f'plot_{j}_{k}.jpg'), dpi=500)
                
power_temporal_weight()
# color_map()