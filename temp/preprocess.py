import pandas as pd
import numpy as np
import pickle

## csv load
# 배터리
BT_1_df = pd.read_csv('/nas/home/tmddnjs3467/time-series/ODE/SCADA_HISTORY_DATA2_6.csv', index_col=0)
BT_2_df = pd.read_csv('/nas/home/tmddnjs3467/time-series/ODE/SCADA_HISTORY_DATA2_7.csv', index_col=0)
BT_3_df = pd.read_csv('/nas/home/tmddnjs3467/time-series/ODE/SCADA_HISTORY_DATA2_8.csv', index_col=0)

# 태양광
PV_df = pd.read_csv('/nas/home/tmddnjs3467/time-series/ODE/SCADA_HISTORY_DATA2_10.csv', index_col=0)

# 끌어다쓰는 전력
ACB_df = pd.read_csv('/nas/home/tmddnjs3467/time-series/ODE/SCADA_HISTORY_DATA2_11.csv', index_col=0)

# 이상치 처리: 65534 값 제거 후 linear interpolate
# 24, 21, 25개
BT_1_df['배터리#1호기.Bank Power'].replace(65534, np.NaN, inplace=True)
BT_2_df['배터리#2호기.Bank Power'].replace(65534, np.NaN, inplace=True)
BT_3_df['배터리#3호기.Bank Power'].replace(65534, np.NaN, inplace=True)

BT_1_df['배터리#1호기.Bank Power'].interpolate(inplace=True)
BT_2_df['배터리#2호기.Bank Power'].interpolate(inplace=True)
BT_3_df['배터리#3호기.Bank Power'].interpolate(inplace=True)

# 3313개
PV_df['태양광시스템.인버터 AC전력'].replace(65534, np.NaN, inplace=True)
PV_df['태양광시스템.인버터 AC전력'].interpolate(inplace=True)

# 29개
ACB_df['ACB 계전기.TOTAL 전력'].replace(65534, np.NaN, inplace=True)
ACB_df['ACB 계전기.TOTAL 전력'].interpolate(inplace=True)

# Bank Current(전류)가 -인 전력 값은 더하고 +인 값은 빼야함
BT_1_df['배터리#1호기.Bank Power'] = -1 * np.sign(BT_1_df['배터리#1호기.BANK Current']) * BT_1_df['배터리#1호기.Bank Power']
BT_2_df['배터리#2호기.Bank Power'] = -1 * np.sign(BT_2_df['배터리#2호기.BANK Current']) * BT_2_df['배터리#2호기.Bank Power']
BT_3_df['배터리#3호기.Bank Power'] = -1 * np.sign(BT_3_df['배터리#3호기.BANK Current']) * BT_3_df['배터리#3호기.Bank Power']

## 데이터 1시간 간격으로 resample
BT_1_df.index = pd.to_datetime(BT_1_df.index)
BT_2_df.index = pd.to_datetime(BT_2_df.index)
BT_3_df.index = pd.to_datetime(BT_3_df.index)

PV_df.index = pd.to_datetime(PV_df.index)

ACB_df.index = pd.to_datetime(ACB_df.index)

BT_1_df_hour = BT_1_df.resample(rule='H').sum()
BT_2_df_hour = BT_2_df.resample(rule='H').sum()
BT_3_df_hour = BT_3_df.resample(rule='H').sum()

PV_df_hour = PV_df.resample(rule='H').sum()

ACB_df_hour = ACB_df.resample(rule='H').sum() / 1000. # w -> kw로 단위 변환

# 종합
concat_df = pd.DataFrame()
concat_df['BT_1'] = BT_1_df_hour['배터리#1호기.Bank Power'].copy()
concat_df['BT_2'] = BT_2_df_hour['배터리#2호기.Bank Power'].copy()
concat_df['BT_3'] = BT_3_df_hour['배터리#3호기.Bank Power'].copy()
concat_df['PV_AC'] = PV_df_hour['태양광시스템.인버터 AC전력'].copy()
concat_df['ACB_TOTAL'] = ACB_df_hour['ACB 계전기.TOTAL 전력'].copy()
with open('./datatime_index.pkl', 'wb') as f:
    pickle.dump(pd.to_datetime(ACB_df_hour.index.values), f)

# 60분 + 5분 전처리
input_length = 72
output_length = 12

data = []
label = []
for i in range(len(concat_df)-(input_length+output_length)+1):
    data.append(np.sum(concat_df.iloc[i:i+input_length].values, axis=1))
    label.append(np.sum(concat_df.iloc[i+input_length:i+(input_length+output_length)].values, axis=1))
data = np.array(data)
label = np.array(label)

# 저장
np.save('./x_total.npy', data)
np.save('./y_total.npy', label)
