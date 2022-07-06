import pandas as pd
import numpy as np
import pickle

BT_1_df = pd.read_csv('/nas/home/tmddnjs3467/time-series/ODE/SCADA_HISTORY_DATA2_6.csv')
BT_2_df = pd.read_csv('/nas/home/tmddnjs3467/time-series/ODE/SCADA_HISTORY_DATA2_7.csv')
BT_3_df = pd.read_csv('/nas/home/tmddnjs3467/time-series/ODE/SCADA_HISTORY_DATA2_8.csv')

# 배터리 이상치 처리
BT_1_df['배터리#1호기.Bank Power'].replace(65534, np.NaN, inplace=True)
BT_2_df['배터리#2호기.Bank Power'].replace(65534, np.NaN, inplace=True)
BT_3_df['배터리#3호기.Bank Power'].replace(65534, np.NaN, inplace=True)

BT_1_df['배터리#1호기.Bank Power'].interpolate(inplace=True)
BT_2_df['배터리#2호기.Bank Power'].interpolate(inplace=True)
BT_3_df['배터리#3호기.Bank Power'].interpolate(inplace=True)

# 배터리 변화량 계산
BT_1_diff = []
BT_2_diff = []
BT_3_diff = []
for i in range(len(BT_1_df)-1):
    BT_1_diff.append(BT_1_df['배터리#1호기.Bank Power'][i+1]-BT_1_df['배터리#1호기.Bank Power'][i])
    BT_2_diff.append(BT_2_df['배터리#2호기.Bank Power'][i+1]-BT_2_df['배터리#2호기.Bank Power'][i])
    BT_3_diff.append(BT_3_df['배터리#3호기.Bank Power'][i+1]-BT_3_df['배터리#3호기.Bank Power'][i])

PV_df = pd.read_csv('/nas/home/tmddnjs3467/time-series/ODE/SCADA_HISTORY_DATA2_10.csv')
# PV 이상치 처리
PV_df['태양광시스템.인버터 AC전력'].replace(65534, np.NaN, inplace=True)
PV_df['태양광시스템.인버터 AC전력'].interpolate(inplace=True)

ACB_df = pd.read_csv('/nas/home/tmddnjs3467/time-series/ODE/SCADA_HISTORY_DATA2_11.csv')
# ACB -값 이상치 처리
ACB_df['ACB 계전기.TOTAL 전력'].loc[ACB_df['ACB 계전기.TOTAL 전력']<0] = 0

concat_df = pd.DataFrame()
concat_df['BT_1'] = BT_1_diff
concat_df['BT_2'] = BT_2_diff
concat_df['BT_3'] = BT_3_diff
concat_df['PV_AC'] = PV_df['태양광시스템.인버터 AC전력'][1:].reset_index(drop=True)
concat_df['ACB_TOTAL'] = ACB_df['ACB 계전기.TOTAL 전력'][1:].reset_index(drop=True)

# 60분 + 5분 전처리
input_length = 60
output_length = 5

data = []
label = []
for i in range(len(concat_df)-(input_length+output_length)+1):
    data.append(concat_df.iloc[i:i+input_length].values)
    label.append(np.sum(concat_df.iloc[i+input_length:i+(input_length+output_length)].values, axis=0))
data = np.array(data)
label = np.array(label)

# 저장
np.save('./x.npy', data)
np.save('./y.npy', label)
