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
            if self.version == 'B':
                self.Seasonal_Linear = nn.Linear(self.seq_len, self.pred_len)
                self.Trend_Linear = nn.Linear(self.seq_len, self.pred_len)

            if self.version == 'TWM-2':
                self.Seasonal_Linear = nn.Linear(self.seq_len, self.pred_len)
                self.Trend_Linear = nn.Linear(self.seq_len, self.pred_len)
                self.Trend_power_Linear = nn.Linear(self.seq_len, self.seq_len)
                self.sigmoid = nn.Sigmoid()
                if self.time == 'cat':
                    self.time_weight_linear = nn.Linear(self.seq_len*4, self.seq_len)          
                if self.time == 'conv':
                    self.time_Conv = nn.Conv1d(in_channels=self.time_channel, out_channels=1, kernel_size=1, bias=False)            
                if self.weight == 'linear':
                    self.time_Linear = nn.Linear(self.seq_len, self.seq_len)
                if self.weight == 'CatLinear':
                    self.time_cat_linear = nn.Linear(self.seq_len*self.time_channel, self.seq_len)
                if self.weight == 'sigmoid':
                    self.sigmoid = nn.Sigmoid()
                
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
            
            if self.version == 'B':
                seasonal_output = self.Seasonal_Linear(seasonal_init.permute(0,2,1)).permute(0,2,1)
                trend_output = self.Trend_Linear(trend_init.permute(0,2,1)).permute(0,2,1)
            
            if self.version == 'TWM-2':
                if self.time == 'cat':
                    time = x_mark.reshape(32,288,1)
                    time = self.time_weight_linear(time.permute(0,2,1)).permute(0,2,1)
                if self.time == 'conv':
                    time = self.time_Conv(x_mark.permute(0,2,1)).permute(0,2,1)
                if self.weight == 'softmax':
                    time = F.softmax(time, dim=1)
                if self.weight == 'sigmoid':
                    time = self.sigmoid(time)
                if self.weight == 'linear':
                    time = self.time_Linear(time.permute(0,2,1)).permute(0,2,1)
                time = self.sigmoid(time)
                trend_init = self.Trend_power_Linear(trend_init.permute(0,2,1)).permute(0,2,1)
                trend_init = trend_init * time
                seasonal_output = self.Seasonal_Linear(seasonal_init.permute(0,2,1)).permute(0,2,1)
                trend_output = self.Trend_Linear(trend_init.permute(0,2,1)).permute(0,2,1)
                
        x = seasonal_output + trend_output
        return x # to [Batch, Output length, Channel]
