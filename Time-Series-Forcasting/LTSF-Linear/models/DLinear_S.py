import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class series_decomp(nn.Module):
    """
    Series decomposition block
    """
    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean

class Model(nn.Module):
    """
    Decomposition-Linear
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len

        # Decompsition Kernel Size
        kernel_size = 25
        self.decompsition = series_decomp(kernel_size)
        self.individual = configs.individual
        self.channels = configs.enc_in
        self.version = configs.ver
        self.weight = configs.weight
        self.time_channel = configs.time_channel
        self.time = configs.time
        self.time_emb = configs.time_emb
        self.emb_dim = configs.emb_dim

        self.data = configs.data
        self.weather = configs.weather
        self.traffic = configs.traffic
        self.electricity = configs.electricity
        self.exchange = configs.exchange

        if self.individual:
            self.Seasonal_Linear = nn.ModuleList()
            seTrend_Linearend = nn.ModuleList()
            
            for i in range(self.channels):
                self.Seasonal_Linear.append(nn.Linear(self.seq_len,self.pred_len))
                seTrend_Linearend.append(nn.Linear(self.seq_len,self.pred_len))

                # Use this two lines if you want to visualize the weights
                # self.Seasonal_Linear[i].weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))
                # seTrend_Linearend[i].weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))
        else:
            if self.time_emb:
                self.month_emb = nn.Embedding(13, self.emb_dim)
                self.day_emb = nn.Embedding(31, self.emb_dim)
                self.weekday_emb = nn.Embedding(7, self.emb_dim)
                if ('ETTm' in self.data) or self.weather:
                    self.minite_emb = nn.Embedding(61, self.emb_dim)
                    self.hour_emb = nn.Embedding(24, self.emb_dim)
                elif self.exchange:
                    pass
                else:
                    self.hour_emb = nn.Embedding(24, self.emb_dim)
            
            if self.version == 'B-oneHot':
                self.Seasonal_Linear = nn.Linear(self.seq_len, self.pred_len)
                self.Trend_Linear = nn.Linear(self.seq_len, self.pred_len)
                self.sigmoid = nn.Sigmoid()
                self.time_Conv = nn.Conv1d(in_channels=self.time_channel, out_channels=1, kernel_size=1, bias=False) # time channel 22         
                self.time_Linear = nn.Linear(self.seq_len, self.seq_len)
                
            if self.version == 'oneHot':
                self.Seasonal_Linear = nn.Linear(self.seq_len, self.pred_len)
                self.Trend_Linear = nn.Linear(self.seq_len, self.pred_len)
                self.power_Seasonal_Linear = nn.Linear(self.seq_len, self.seq_len)
                self.sigmoid = nn.Sigmoid()
                self.time_Conv = nn.Conv1d(in_channels=self.time_channel, out_channels=1, kernel_size=1, bias=False) # time channel 22         
                self.time_Linear = nn.Linear(self.seq_len, self.seq_len)
                
            if self.version == 'oneHot-ReLU':
                self.Seasonal_Linear = nn.Linear(self.seq_len, self.pred_len)
                self.Trend_Linear = nn.Linear(self.seq_len, self.pred_len)
                self.power_Seasonal_Linear = nn.Linear(self.seq_len, self.seq_len)
                self.sigmoid = nn.Sigmoid()
                self.relu = nn.ReLU()
                self.time_Conv = nn.Conv1d(in_channels=self.time_channel, out_channels=1, kernel_size=1, bias=False) # time channel 22         
                self.time_Linear = nn.Linear(self.seq_len, self.seq_len)
            
            if self.version == 'B':
                self.Seasonal_Linear = nn.Linear(self.seq_len, self.pred_len)
                self.Trend_Linear = nn.Linear(self.seq_len, self.pred_len)

            if self.version == 'D':
                self.Seasonal_Linear1 = nn.Linear(self.seq_len, 72)
                self.Trend_Linear1 = nn.Linear(self.seq_len, 72)
                self.Seasonal_Linear2 = nn.Linear(72, self.pred_len)
                self.Trend_Linear2 = nn.Linear(72, self.pred_len)
                
            if self.version == 'D-ReLU':
                self.Seasonal_Linear1 = nn.Linear(self.seq_len, 72)
                self.Trend_Linear1 = nn.Linear(self.seq_len, 72)
                self.Seasonal_Linear2 = nn.Linear(72, self.pred_len)
                self.Trend_Linear2 = nn.Linear(72, self.pred_len)
                self.relu = nn.ReLU()
            
            if self.version == 'B-TWM':
                self.Seasonal_Linear = nn.Linear(self.seq_len, self.pred_len)
                self.Trend_Linear = nn.Linear(self.seq_len, self.pred_len)
                self.sigmoid = nn.Sigmoid()
                self.time_Conv = nn.Conv1d(in_channels=self.time_channel, out_channels=1, kernel_size=1, bias=False)            
                self.time_Linear = nn.Linear(self.seq_len, self.seq_len)
            
            if self.version == 'TWM':
                self.Seasonal_Linear = nn.Linear(self.seq_len, self.pred_len)
                self.Trend_Linear = nn.Linear(self.seq_len, self.pred_len)
                self.sigmoid = nn.Sigmoid()
                self.time_Conv = nn.Conv1d(in_channels=self.time_channel, out_channels=1, kernel_size=1, bias=False)            
                self.time_Linear = nn.Linear(self.seq_len, self.seq_len)
                self.power_Seasonal_Linear = nn.Linear(self.seq_len, self.seq_len)
                
            if self.version == 'TWM-ReLU':
                self.Seasonal_Linear = nn.Linear(self.seq_len, self.pred_len)
                self.Trend_Linear = nn.Linear(self.seq_len, self.pred_len)
                self.relu = nn.ReLU()
                self.sigmoid = nn.Sigmoid()
                self.time_Conv = nn.Conv1d(in_channels=self.time_channel, out_channels=1, kernel_size=1, bias=False)            
                self.time_Linear = nn.Linear(self.seq_len, self.seq_len)
                self.power_Seasonal_Linear = nn.Linear(self.seq_len, self.seq_len)
                
    def forward(self, x, x_mark):
        # x: [Batch, Input length, Channel]
        seasonal_init, trend_init = self.decompsition(x)

        if self.individual:
            seasonal_output = torch.zeros([seasonal_init.size(0),seasonal_init.size(1),self.pred_len],dtype=seasonal_init.dtype).to(seasonal_init.device)
            trend_output = torch.zeros([trend_init.size(0),trend_init.size(1),self.pred_len],dtype=trend_init.dtype).to(trend_init.device)
            for i in range(self.channels):
                seasonal_output[:,i,:] = self.Seasonal_Linear[i](seasonal_init[:,i,:])
                trend_output[:,i,:] = self.Trend_Linear[i](trend_init[:,i,:])
        else:
            if self.time_emb: # 일반 TWM으로 해야해
                x_mark = x_mark.long()
                x_mark_month = self.month_emb(x_mark[:,:,0]-1) # 월
                x_mark_day = self.day_emb(x_mark[:,:,1]-1) # 일
                x_mark_weekday = self.weekday_emb(x_mark[:,:,2]) # 요일
                if ('ETTm' in self.data) or self.weather:
                    x_mark_hour = self.hour_emb(x_mark[:,:,3]) # 시간
                    x_mark_min = self.minite_emb(x_mark[:,:,4])
                    x_mark = torch.cat([x_mark_month, x_mark_day, x_mark_weekday, x_mark_hour, x_mark_min], dim=2)
                elif self.exchange:
                    x_mark = torch.cat([x_mark_month, x_mark_day, x_mark_weekday], dim=2)
                else:
                    x_mark_hour = self.hour_emb(x_mark[:,:,3]) # 시간
                    x_mark = torch.cat([x_mark_month, x_mark_day, x_mark_weekday, x_mark_hour], dim=2)

            if self.version == 'B-oneHot':
                x_mark = x[:,:,1:] # 전력 제외 one hot vector
                time = self.time_Conv(x_mark.permute(0,2,1)).permute(0,2,1)
                time = self.time_Linear(time.permute(0,2,1)).permute(0,2,1)
                time = self.sigmoid(time)
                seasonal_init = seasonal_init*time 
                seasonal_output = self.Seasonal_Linear(seasonal_init.permute(0,2,1)).permute(0,2,1)
                trend_output = self.Trend_Linear(trend_init.permute(0,2,1)).permute(0,2,1)
            
            if self.version == 'oneHot':
                x_mark = x[:,:,1:] # 전력 제외 one hot vector
                time = self.time_Conv(x_mark.permute(0,2,1)).permute(0,2,1)
                time = self.time_Linear(time.permute(0,2,1)).permute(0,2,1)
                time = self.sigmoid(time)
                seasonal_init = self.power_Seasonal_Linear(seasonal_init.permute(0,2,1)).permute(0,2,1)
                seasonal_init = seasonal_init*time 
                seasonal_output = self.Seasonal_Linear(seasonal_init.permute(0,2,1)).permute(0,2,1)
                trend_output = self.Trend_Linear(trend_init.permute(0,2,1)).permute(0,2,1)
                
            if self.version == 'oneHot-ReLU':
                x_mark = x[:,:,1:] # 전력 제외 one hot vector
                time = self.time_Conv(x_mark.permute(0,2,1)).permute(0,2,1)
                time = self.time_Linear(time.permute(0,2,1)).permute(0,2,1)
                time = self.sigmoid(time)
                seasonal_init = self.power_Seasonal_Linear(seasonal_init.permute(0,2,1)).permute(0,2,1)
                seasonal_init = self.relu(seasonal_init)
                seasonal_init = seasonal_init*time 
                seasonal_output = self.Seasonal_Linear(seasonal_init.permute(0,2,1)).permute(0,2,1)
                trend_output = self.Trend_Linear(trend_init.permute(0,2,1)).permute(0,2,1)
               
            if self.version == 'B':
                seasonal_output = self.Seasonal_Linear(seasonal_init.permute(0,2,1)).permute(0,2,1)
                trend_output = self.Trend_Linear(trend_init.permute(0,2,1)).permute(0,2,1)
            
            if self.version =='D':
                seasonal_init = self.Seasonal_Linear1(seasonal_init.permute(0,2,1)).permute(0,2,1)
                trend_init = self.Trend_Linear1(trend_init.permute(0,2,1)).permute(0,2,1)
                seasonal_output = self.Seasonal_Linear2(seasonal_init.permute(0,2,1)).permute(0,2,1)
                trend_output = self.Trend_Linear2(trend_init.permute(0,2,1)).permute(0,2,1)   
            
            if self.version =='D-ReLU':
                seasonal_init = self.Seasonal_Linear1(seasonal_init.permute(0,2,1)).permute(0,2,1)
                trend_init = self.Trend_Linear1(trend_init.permute(0,2,1)).permute(0,2,1)
                seasonal_init = self.relu(seasonal_init)
                trend_init = self.relu(trend_init)
                seasonal_output = self.Seasonal_Linear2(seasonal_init.permute(0,2,1)).permute(0,2,1)
                trend_output = self.Trend_Linear2(trend_init.permute(0,2,1)).permute(0,2,1) 
            
            if self.version =='B-TWM':
                time = self.time_Conv(x_mark.permute(0,2,1)).permute(0,2,1)
                time = self.time_Linear(time.permute(0,2,1)).permute(0,2,1)
                time = self.sigmoid(time)
                seasonal_init = seasonal_init * time
                seasonal_output = self.Seasonal_Linear(seasonal_init.permute(0,2,1)).permute(0,2,1)
                trend_output = self.Trend_Linear(trend_init.permute(0,2,1)).permute(0,2,1)
                
            if self.version == 'TWM':
                time = self.time_Conv(x_mark.permute(0,2,1)).permute(0,2,1)
                time = self.time_Linear(time.permute(0,2,1)).permute(0,2,1)
                time = self.sigmoid(time)
                seasonal_init = self.power_Seasonal_Linear(seasonal_init.permute(0,2,1)).permute(0,2,1)
                seasonal_init = seasonal_init * time
                seasonal_output = self.Seasonal_Linear(seasonal_init.permute(0,2,1)).permute(0,2,1)
                trend_output = self.Trend_Linear(trend_init.permute(0,2,1)).permute(0,2,1)

            if self.version == 'TWM-ReLU':
                time = self.time_Conv(x_mark.permute(0,2,1)).permute(0,2,1)
                time = self.time_Linear(time.permute(0,2,1)).permute(0,2,1)
                time = self.sigmoid(time)    
                seasonal_init = self.power_Seasonal_Linear(seasonal_init.permute(0,2,1)).permute(0,2,1)
                seasonal_init = self.relu(seasonal_init)
                seasonal_init = seasonal_init * time
                seasonal_output = self.Seasonal_Linear(seasonal_init.permute(0,2,1)).permute(0,2,1)
                trend_output = self.Trend_Linear(trend_init.permute(0,2,1)).permute(0,2,1)

        x = seasonal_output + trend_output
        return x # to [Batch, Output length, Channel]
