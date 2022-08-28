import pandas as pd
import numpy as np
import pickle

SAVEDIR_PATH = './'
INPUT_LENGTH = 72
OUTPUT_LENGTH = 12
INTERVAL_UNIT = 'H' # H or 30S (시간별로 또는 30초별로)

## csv load
# 배터리
BT_1_df = pd.read_csv('/nas/datahub/mirae/Data/BT_1_except_NaN.csv', index_col=0)
BT_2_df = pd.read_csv('/nas/datahub/mirae/Data/BT_2_except_NaN.csv', index_col=0)
BT_3_df = pd.read_csv('/nas/datahub/mirae/Data/BT_3_except_NaN.csv', index_col=0)

# 태양광
PV_df = pd.read_csv('/nas/datahub/mirae/Data/PV_except_NaN.csv', index_col=0)

# 끌어다쓰는 전력
ACB_df = pd.read_csv('/nas/datahub/mirae/Data/ACB_except_NaN.csv', index_col=0)

# df들의 index를 datetime으로 type 변경
BT_1_df.index = pd.to_datetime(BT_1_df.index)
BT_2_df.index = pd.to_datetime(BT_2_df.index)
BT_3_df.index = pd.to_datetime(BT_3_df.index)

PV_df.index = pd.to_datetime(PV_df.index)

ACB_df.index = pd.to_datetime(ACB_df.index)

## 모든 csv의 시작 시점이 동일하도록 맞추기
min_date = ACB_df.index.min()
for i in [BT_1_df, BT_2_df, BT_3_df, PV_df]:
    if i.index.min() > min_date:
        min_date = i.index.min()

if min_date.hour != 0 or min_date.minute != 0 or min_date.second != 0:
    min_date = pd.to_datetime(f'{min_date.year}-{min_date.month}-{min_date.day+1} 00:00:00')
BT_1_df = BT_1_df[BT_1_df.index > min_date]
BT_2_df = BT_2_df[BT_2_df.index > min_date]
BT_3_df = BT_3_df[BT_3_df.index > min_date]
PV_df = PV_df[PV_df.index > min_date]
ACB_df = ACB_df[ACB_df.index > min_date]
    
# 이상치 처리: 65534 값 제거 후 linear interpolate
BT_1_df['배터리#1호기.Bank Power'].replace(65534, np.NaN, inplace=True)
BT_2_df['배터리#2호기.Bank Power'].replace(65534, np.NaN, inplace=True)
BT_3_df['배터리#3호기.Bank Power'].replace(65534, np.NaN, inplace=True)

BT_1_df['배터리#1호기.Bank Power'].interpolate(inplace=True)
BT_2_df['배터리#2호기.Bank Power'].interpolate(inplace=True)
BT_3_df['배터리#3호기.Bank Power'].interpolate(inplace=True)

PV_df['태양광시스템.인버터 AC전력'].replace(65534, np.NaN, inplace=True)
PV_df['태양광시스템.인버터 AC전력'].interpolate(inplace=True)

ACB_df['ACB 계전기.TOTAL 전력'].replace(65534, np.NaN, inplace=True)
ACB_df['ACB 계전기.TOTAL 전력'].interpolate(inplace=True)

# Bank Current(전류)가 -인 전력 값은 더하고 +인 값은 빼야함
BT_1_df['배터리#1호기.Bank Power'] = -1 * np.sign(BT_1_df['배터리#1호기.BANK Current']) * BT_1_df['배터리#1호기.Bank Power']
BT_2_df['배터리#2호기.Bank Power'] = -1 * np.sign(BT_2_df['배터리#2호기.BANK Current']) * BT_2_df['배터리#2호기.Bank Power']
BT_3_df['배터리#3호기.Bank Power'] = -1 * np.sign(BT_3_df['배터리#3호기.BANK Current']) * BT_3_df['배터리#3호기.Bank Power']

## 데이터 1시간 간격으로 resample
if INTERVAL_UNIT == 'H':
    BT_1_df_re = BT_1_df.resample(rule=INTERVAL_UNIT, origin='end', closed='right').mean()
    BT_2_df_re = BT_2_df.resample(rule=INTERVAL_UNIT, origin='end', closed='right').mean()
    BT_3_df_re = BT_3_df.resample(rule=INTERVAL_UNIT, origin='end', closed='right').mean()

    PV_df_re = PV_df.resample(rule=INTERVAL_UNIT, origin='end', closed='right').mean()

    ACB_df_re = ACB_df.resample(rule=INTERVAL_UNIT, origin='end', closed='right').mean() / 1000. # w -> kw로 단위 변환
else:
    BT_1_df_re = BT_1_df.resample(rule=INTERVAL_UNIT, origin='end', closed='right').last()
    BT_2_df_re = BT_2_df.resample(rule=INTERVAL_UNIT, origin='end', closed='right').last()
    BT_3_df_re = BT_3_df.resample(rule=INTERVAL_UNIT, origin='end', closed='right').last()

    PV_df_re = PV_df.resample(rule=INTERVAL_UNIT, origin='end', closed='right').last()

    ACB_df_re = ACB_df.resample(rule=INTERVAL_UNIT, origin='end', closed='right').last() / 1000. # w -> kw로 단위 변환

# 종합
concat_df = pd.DataFrame()
concat_df.index = BT_1_df_re.index
concat_df['BT_1'] = BT_1_df_re['배터리#1호기.Bank Power']
concat_df['BT_2'] = BT_2_df_re['배터리#2호기.Bank Power']
concat_df['BT_3'] = BT_3_df_re['배터리#3호기.Bank Power']
concat_df['PV_AC'] = PV_df_re['태양광시스템.인버터 AC전력']
concat_df['ACB_TOTAL'] = ACB_df_re['ACB 계전기.TOTAL 전력']
concat_df['TOTAL'] = np.sum(concat_df.values, axis=1)

concat_df.to_csv(f'./total_{INTERVAL_UNIT}.csv')


# 전처리
data = []
label = []
time = []
for i in range(len(concat_df)-(INPUT_LENGTH+OUTPUT_LENGTH)+1):
    if np.sum(concat_df.iloc[i:i+(INPUT_LENGTH+OUTPUT_LENGTH)]['TOTAL'] == 0) == 0: # 범위 안에 결측치가 없을 때만 데이터 추출
        data.append(concat_df.iloc[i:i+INPUT_LENGTH]['TOTAL'])
        label.append(concat_df.iloc[i+INPUT_LENGTH:i+(INPUT_LENGTH+OUTPUT_LENGTH)]['TOTAL'])
        time.append(concat_df.index[i:i+(INPUT_LENGTH+OUTPUT_LENGTH)].values)
data = np.array(data)
label = np.array(label)
time = np.array(time)

# 저장
np.save(f'./x_{INTERVAL_UNIT}_total.npy', data)
np.save(f'./y_{INTERVAL_UNIT}_total.npy', label)
np.save(f'./time_{INTERVAL_UNIT}_index.npy', time)
