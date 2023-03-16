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

class DLinear(nn.Module):
    """
    Decomposition-Linear
    """
    def __init__(self, seq_len, pred_len):
        super(DLinear, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len

        # Decompsition Kernel Size
        kernel_size = 25
        self.decompsition = series_decomp(kernel_size)
        
        self.Seasonal_Linear = nn.Linear(self.seq_len, self.pred_len)
        self.Trend_Linear = nn.Linear(self.seq_len, self.pred_len)
                
    def forward(self, x, x_mark):
        # x: [Batch, Input length, Channel]
        seasonal_init, trend_init = self.decompsition(x)
        seasonal_output = self.Seasonal_Linear(seasonal_init.permute(0,2,1)).permute(0,2,1)
        trend_output = self.Trend_Linear(trend_init.permute(0,2,1)).permute(0,2,1)
        x = seasonal_output + trend_output
        return x # to [Batch, Output length, Channel]

class betterDLinear(nn.Module):
    def __init__(self, seq_len, pred_len):
        super(betterDLinear, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.Seasonal_Linear = nn.Linear(self.seq_len, self.pred_len)
        self.Trend_Linear = nn.Linear(self.seq_len, self.pred_len)
        self.power_Seasonal_Linear = nn.Linear(self.seq_len, self.seq_len)
        self.power_Trend_Linear = nn.Linear(self.seq_len, self.seq_len)
        self.sigmoid = nn.Sigmoid()
        self.time_Conv = nn.Conv1d(in_channels=4, out_channels=1, kernel_size=1, bias=False)            
        self.time_Linear = nn.Linear(self.seq_len, self.seq_len)
        kernel_size = 25
        self.decompsition = series_decomp(kernel_size)

    def forward(self, x, x_mark):
        seasonal_init, trend_init = self.decompsition(x)
        time = self.time_Conv(x_mark.permute(0,2,1)).permute(0,2,1)
        time = self.time_Linear(time.permute(0,2,1)).permute(0,2,1)
        time = self.sigmoid(time)
        seasonal_init = self.power_Seasonal_Linear(seasonal_init.permute(0,2,1)).permute(0,2,1)
        trend_init = self.power_Trend_Linear(trend_init.permute(0,2,1)).permute(0,2,1)
        seasonal_init = seasonal_init * time
        trend_init = trend_init * time
        seasonal_output = self.Seasonal_Linear(seasonal_init.permute(0,2,1)).permute(0,2,1)
        trend_output = self.Trend_Linear(trend_init.permute(0,2,1)).permute(0,2,1)
        x = seasonal_output + trend_output
        return x