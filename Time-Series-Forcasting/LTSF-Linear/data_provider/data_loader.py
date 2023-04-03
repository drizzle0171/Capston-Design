import os
import numpy as np
import pandas as pd
import os
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from utils.timefeatures import time_features
import warnings
import pickle

warnings.filterwarnings('ignore')

class Dataset_Mirae_H(Dataset):
    def __init__(self, dataset_type, time_emb, weather, traffic, electricity, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h', rep=None):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.time_emb = time_emb
        self.dataset_type = dataset_type
        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        
        # 22.02.19 ~ 22.7.1 00:00:00
        x_origin = np.load('/nas/home/drizzle0171/Time-Series-Forcasting/mirea-share/preprocessing/x_final_origin.npy')
        y_origin = np.load('/nas/home/drizzle0171/Time-Series-Forcasting/mirea-share/preprocessing/y_final_origin.npy')
        time_origin = np.load('/nas/home/drizzle0171/Time-Series-Forcasting/mirea-share/preprocessing/time_index_final_origin.npy')
        
        # 22.7.1 01:00:00 ~ 22.8.1 00:00:00
        x_227 = np.load('/nas/home/drizzle0171/Time-Series-Forcasting/mirea-share/preprocessing/x_final_227.npy')
        y_227 = np.load('/nas/home/drizzle0171/Time-Series-Forcasting/mirea-share/preprocessing/y_final_227.npy')
        time_227 = np.load('/nas/home/drizzle0171/Time-Series-Forcasting/mirea-share/preprocessing/time_index_final_227.npy')
 
        # 22.8.1 ~ 23.2.1
        x_new = np.load('/nas/home/drizzle0171/Time-Series-Forcasting/mirea-share/preprocessing/x_final_new.npy')
        y_new = np.load('/nas/home/drizzle0171/Time-Series-Forcasting/mirea-share/preprocessing/y_final_new.npy')
        time_new = np.load('/nas/home/drizzle0171/Time-Series-Forcasting/mirea-share/preprocessing/time_index_final_new.npy')

        # 23.2.1 ~ 23.3.25
        x_recent = np.load('/nas/home/drizzle0171/Time-Series-Forcasting/mirea-share/preprocessing/x_final_recent.npy')
        y_recent = np.load('/nas/home/drizzle0171/Time-Series-Forcasting/mirea-share/preprocessing/y_final_recent.npy')
        time_recent = np.load('/nas/home/drizzle0171/Time-Series-Forcasting/mirea-share/preprocessing/time_index_final_recent.npy')

        x = np.concatenate((x_origin, x_227, x_new, x_recent), axis=0)
        y = np.concatenate((y_origin, y_227, y_new, y_recent), axis=0)
        time = np.concatenate((time_origin, time_227, time_new, time_recent), axis=0)
        
        train_data = x[:int(len(x)*0.7)]
        max_x = np.max(train_data)
        min_x = np.min(train_data)
        x = (x - min_x) / (max_x - min_x)
        y = (y - min_x) / (max_x - min_x)
        
        # temp_x = x
        # temp_y = y
        # temp_x_test = x_test
        # temp_y_test = y_test
        
        # for normalization for oneHot encoding
        # train_data = x[:int(len(x)*0.7)]
        # max_x = np.max(np.max(train_data, axis=1), axis=0)[0]
        # min_x = np.min(np.min(train_data, axis=1), axis=0)[0]
        # x = np.zeros(x.shape)
        # y = np.zeros(y.shape)
        # x[:,:,1:] = temp_x[:,:,1:]        
        # y[:,:,1:] = temp_y[:,:,1:]
        # norm_x = (temp_x - min_x) / (max_x - min_x)
        # norm_y = (temp_y - min_x) / (max_x - min_x)
        # x[:,:,0] = norm_x[:,:,0]
        # y[:,:,0] = norm_y[:,:,0]

        # x_test = np.zeros(x_test.shape)
        # y_test = np.zeros(y_test.shape)
        # x_test[:,:,1:] = temp_x_test[:,:,1:]        
        # y_test[:,:,1:] = temp_y_test[:,:,1:]
        # norm_x = (temp_x_test - min_x) / (max_x - min_x)
        # norm_y = (temp_y_test - min_x) / (max_x - min_x)
        # x_test[:,:,0] = norm_x[:,:,0]
        # y_test[:,:,0] = norm_y[:,:,0]
        if self.dataset_type == 'b70b15brest': # 기존 데이터 70 학습 + 기존 데이터 15 검증 + 기존 데이터 나머지 모두 테스트
            num_train = int(len(x)*0.7)
            num_val_85 = int(len(x)*0.85)
            border1s = [0, num_train, num_val_85] 
            border2s = [num_train, num_val_85, len(x)]
            border1 = border1s[self.set_type]
            border2 = border2s[self.set_type]
            
            self.data_x = x[border1:border2].reshape(-1, x.shape[1], 1)
            self.data_y = y[border1:border2].reshape(-1, y.shape[1], 1)
            df_stamp = time[border1:border2]
            
        if self.dataset_type == 'b70b15orest': # 기존 데이터 70 학습 + 기존 데이터 15 검증 + 새로운 데이터 모두 테스트
            num_train = int(len(x)*0.7)
            num_val_85 = int(len(x)*0.85)
            border1s = [0, num_train, 0] 
            border2s = [num_train, num_val_85, len(x_test)]
            border1 = border1s[self.set_type]
            border2 = border2s[self.set_type]
            
            if self.set_type == 2:
                self.data_x = x_test[border1:border2].reshape(-1, x.shape[1], 1)
                self.data_y = y_test[border1:border2].reshape(-1, y.shape[1], 1)
                df_stamp = test_time[border1:border2]
            else:
                self.data_x = x[border1:border2].reshape(-1, x.shape[1], 1)
                self.data_y = y[border1:border2].reshape(-1, y.shape[1], 1)
                df_stamp = time[border1:border2]
            
        if self.dataset_type == 'b70o15orest': # 기존 데이터 70 학습 + 새로운 데이터 15 검증 + 새로운 데이터 나머지 모두 테스트
            num_train = int(len(x)*0.7)
            num_val_15 = int(len(x_test)*0.15)
            border1s = [0, 0, num_val_15] 
            border2s = [num_train, num_val_15, len(x_test)]
            border1 = border1s[self.set_type]
            border2 = border2s[self.set_type]
            
            if self.set_type == 1 or self.set_type == 2:
                self.data_x = x_test[border1:border2].reshape(-1, x.shape[1], 1)
                self.data_y = y_test[border1:border2].reshape(-1, y.shape[1], 1)
                df_stamp = test_time[border1:border2]
            else:
                self.data_x = x[border1:border2].reshape(-1, x.shape[1], 1)
                self.data_y = y[border1:border2].reshape(-1, y.shape[1], 1)
                df_stamp = time[border1:border2]
          
        if self.dataset_type == 'b100o15orest': # 기존 데이터 70 학습 + 기존 데이터 15 검증 + 새로운 데이터 모두 테스트
            num_val_15 = int(len(x_test)*0.15)
            border1s = [0, 0, num_val_15] 
            border2s = [len(x), num_val_15, len(x_test)]
            border1 = border1s[self.set_type]
            border2 = border2s[self.set_type]
            
            if self.set_type == 1 or self.set_type == 2:
                self.data_x = x_test[border1:border2].reshape(-1, x.shape[1], 1)
                self.data_y = y_test[border1:border2].reshape(-1, y.shape[1], 1)
                df_stamp = test_time[border1:border2]
            else:
                self.data_x = x[border1:border2].reshape(-1, x.shape[1], 1)
                self.data_y = y[border1:border2].reshape(-1, y.shape[1], 1)
                df_stamp = time[border1:border2]

        if self.time_emb:
            month = np.array(list(map(lambda x: pd.to_datetime(x).month, df_stamp))).reshape(df_stamp.shape[0],df_stamp.shape[1],1)
            day = np.array(list(map(lambda x: pd.to_datetime(x).day, df_stamp))).reshape(df_stamp.shape[0],df_stamp.shape[1],1)
            weekday = np.array(list(map(lambda x: pd.to_datetime(x).weekday, df_stamp))).reshape(df_stamp.shape[0],df_stamp.shape[1],1)
            hour = np.array(list(map(lambda x: pd.to_datetime(x).hour, df_stamp))).reshape(df_stamp.shape[0],df_stamp.shape[1],1)
        
        else:
            month = (np.array(list(map(lambda x: pd.to_datetime(x).month, df_stamp))).reshape(df_stamp.shape[0],df_stamp.shape[1],1)-1)/11
            day = (np.array(list(map(lambda x: pd.to_datetime(x).day, df_stamp))).reshape(df_stamp.shape[0],df_stamp.shape[1],1)-1)/30
            weekday = np.array(list(map(lambda x: pd.to_datetime(x).weekday, df_stamp))).reshape(df_stamp.shape[0],df_stamp.shape[1],1)/6
            hour = np.array(list(map(lambda x: pd.to_datetime(x).hour, df_stamp))).reshape(df_stamp.shape[0],df_stamp.shape[1],1)/23
            # month = np.array(list(map(lambda x: pd.to_datetime(x).month, df_stamp))).reshape(df_stamp.shape[0],df_stamp.shape[1],1)
            # day = np.array(list(map(lambda x: pd.to_datetime(x).day, df_stamp))).reshape(df_stamp.shape[0],df_stamp.shape[1],1)
            # weekday = np.array(list(map(lambda x: pd.to_datetime(x).weekday, df_stamp))).reshape(df_stamp.shape[0],df_stamp.shape[1],1)
            # hour = np.array(list(map(lambda x: pd.to_datetime(x).hour, df_stamp))).reshape(df_stamp.shape[0],df_stamp.shape[1],1)
        
        data_stamp = np.concatenate((month,day,weekday,hour),axis=-1)
        self.data_stamp = data_stamp

        
    def __getitem__(self, index):
        s_end = self.seq_len
        r_end = self.seq_len + self.pred_len

        seq_x = self.data_x[index]
        seq_y = self.data_y[index]
        seq_x_mark = self.data_stamp[index][:s_end]
        seq_y_mark = self.data_stamp[index][s_end:r_end]
        
        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x)

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

class Dataset_ETT_hour(Dataset):
    def __init__(self, dataset_type, time_emb, weather, traffic, electricity, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h'):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.time_emb = time_emb
        
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        border1s = [0, 12 * 30 * 24 - self.seq_len, 12 * 30 * 24 + 4 * 30 * 24 - self.seq_len]
        border2s = [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values
        
        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        
        if self.time_emb:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], 1).values
        else:
            df_stamp['month'] = (df_stamp.date.apply(lambda row: row.month, 1)-1)/11
            df_stamp['day'] = (df_stamp.date.apply(lambda row: row.day, 1)-1)/30
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)/6
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)/23
            data_stamp = df_stamp.drop(['date'], 1).values    
            
        if self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]
        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

# class Dataset_ETT_hour(Dataset):
#     def __init__(self, root_path, flag='train', size=None,
#                  features='S', data_path='ETTh1.csv',
#                  target='OT', scale=True, timeenc=0, freq='h'):
#         # size [seq_len, label_len, pred_len]
#         # info
#         if size == None:
#             self.seq_len = 24 * 4 * 4
#             self.label_len = 24 * 4
#             self.pred_len = 24 * 4
#         else:
#             self.seq_len = size[0]
#             self.label_len = size[1]
#             self.pred_len = size[2]
#         # init
#         assert flag in ['train', 'test', 'val']
#         type_map = {'train': 0, 'val': 1, 'test': 2}
#         self.set_type = type_map[flag]

#         self.features = features
#         self.target = target
#         self.scale = scale
#         self.timeenc = 0
#         self.freq = freq

#         self.root_path = root_path
#         self.data_path = data_path
#         self.__read_data__()

#     def __read_data__(self):
#         self.scaler = StandardScaler()
#         df_raw = pd.read_csv(os.path.join(self.root_path,
#                                           self.data_path))

#         border1s = [0, 12 * 30 * 24 - self.seq_len, 12 * 30 * 24 + 4 * 30 * 24 - self.seq_len]
#         border2s = [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]
#         border1 = border1s[self.set_type]
#         border2 = border2s[self.set_type]

#         if self.features == 'M' or self.features == 'MS':
#             cols_data = df_raw.columns[1:]
#             df_data = df_raw[cols_data]
#         elif self.features == 'S':
#             df_data = df_raw[[self.target]]

#         if self.scale:
#             train_data = df_data[border1s[0]:border2s[0]]
#             self.scaler.fit(train_data.values)
#             data = self.scaler.transform(df_data.values)
#         else:
#             data = df_data.values
        
#         df_stamp = df_raw[['date']][border1:border2]
#         df_stamp['date'] = pd.to_datetime(df_stamp.date)
        
#         if self.timeenc == 0:
#             df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
#             df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
#             df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
#             df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
#             data_stamp = df_stamp.drop(['date'], 1).values
            
#         elif self.timeenc == 1:
#             data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
#             data_stamp = data_stamp.transpose(1, 0)

#         self.data_x = data[border1:border2]
#         self.data_y = data[border1:border2]
#         self.data_stamp = data_stamp

#     def __getitem__(self, index):
#         s_begin = index
#         s_end = s_begin + self.seq_len
#         r_begin = s_end - self.label_len
#         r_end = r_begin + self.label_len + self.pred_len

#         seq_x = self.data_x[s_begin:s_end]
#         seq_y = self.data_y[r_begin:r_end]
#         seq_x_mark = self.data_stamp[s_begin:s_end]
#         seq_y_mark = self.data_stamp[r_begin:r_end]
#         return seq_x, seq_y, seq_x_mark, seq_y_mark

#     def __len__(self):
#         return len(self.data_x) - self.seq_len - self.pred_len + 1

#     def inverse_transform(self, data):
#         return self.scaler.inverse_transform(data)

class Dataset_ETT_minute(Dataset):
    def __init__(self, dataset_type, time_emb, weather, traffic, electricity, root_path, flag='train', size=None,
                 features='S', data_path='ETTm1.csv',
                 target='OT', scale=True, timeenc=0, freq='t'):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = 0
        self.time_emb = time_emb
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        border1s = [0, 12 * 30 * 24 * 4 - self.seq_len, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4 - self.seq_len]
        border2s = [12 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 8 * 30 * 24 * 4]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.time_emb:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute, 1)
            # df_stamp['minute'] = df_stamp.minute.map(lambda x: x // 15)
            data_stamp = df_stamp.drop(['date'], 1).values
        else:
            df_stamp['month'] = (df_stamp.date.apply(lambda row: row.month, 1)-1)/11
            df_stamp['day'] = (df_stamp.date.apply(lambda row: row.day, 1)-1)/30
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)/6
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)/23
            df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute, 1)/59
            # df_stamp['minute'] = df_stamp.minute.map(lambda x: x // 15)
            data_stamp = df_stamp.drop(['date'], 1).values            
        
        if self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp
        
    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_Custom(Dataset):
    def __init__(self, dataset_type, time_emb, weather, traffic, electricity, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h'):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = 0
        self.freq = freq
        self.time_emb = time_emb
        
        self.weather = weather
        self.traffic = traffic
        self.electricity = electricity
        
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        cols = list(df_raw.columns)
        cols.remove(self.target)
        cols.remove('date')
        df_raw = df_raw[['date'] + cols + [self.target]]
        # print(cols)
        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            # print(self.scaler.mean_)
            # exit()
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        
        if self.time_emb == 1:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            
            if self.traffic:
                df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            if self.electricity:
                df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            if self.weather:
                df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
                df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute, 1)
            data_stamp = df_stamp.drop(['date'], 1).values
        else:
            df_stamp['month'] = (df_stamp.date.apply(lambda row: row.month, 1)-1)/11
            df_stamp['day'] = (df_stamp.date.apply(lambda row: row.day, 1)-1)/30
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)/6
            
            if self.traffic:
                df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)/23
            if self.electricity:
                df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)/23
            if self.weather:
                df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)/23
                df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute, 1)/59
            data_stamp = df_stamp.drop(['date'], 1).values
        
        if self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
    

class Dataset_Pred(Dataset):
    def __init__(self, root_path, flag='pred', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, inverse=False, timeenc=0, freq='15min', cols=None):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['pred']

        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.freq = freq
        self.cols = cols
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))
        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        if self.cols:
            cols = self.cols.copy()
            cols.remove(self.target)
        else:
            cols = list(df_raw.columns)
            cols.remove(self.target)
            cols.remove('date')
        df_raw = df_raw[['date'] + cols + [self.target]]
        border1 = len(df_raw) - self.seq_len
        border2 = len(df_raw)

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            self.scaler.fit(df_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        tmp_stamp = df_raw[['date']][border1:border2]
        tmp_stamp['date'] = pd.to_datetime(tmp_stamp.date)
        pred_dates = pd.date_range(tmp_stamp.date.values[-1], periods=self.pred_len + 1, freq=self.freq)

        df_stamp = pd.DataFrame(columns=['date'])
        df_stamp.date = list(tmp_stamp.date.values) + list(pred_dates[1:])
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute, 1)
            df_stamp['minute'] = df_stamp.minute.map(lambda x: x // 15)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        if self.inverse:
            self.data_y = df_data.values[border1:border2]
        else:
            self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        if self.inverse:
            seq_y = self.data_x[r_begin:r_begin + self.label_len]
        else:
            seq_y = self.data_y[r_begin:r_begin + self.label_len]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
