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
                
            if self.version == 'TLC':
                self.Seasonal_Linear = nn.Linear(self.seq_len*2, self.pred_len)
                self.Trend_Linear = nn.Linear(self.seq_len*2, self.pred_len)
                self.time_Linear = nn.Linear(72*4, 72, bias=False)
            
            if self.version == 'D':
                self.Seasonal_Linear1 = nn.Linear(self.seq_len, 72)
                self.Trend_Linear1 = nn.Linear(self.seq_len, 72)
                self.Seasonal_Linear2 = nn.Linear(72, self.pred_len)
                self.Trend_Linear2 = nn.Linear(72, self.pred_len)
                self.relu = nn.ReLU()

            if self.version == 'TLS':
                self.Seasonal_Linear = nn.Linear(self.seq_len, self.pred_len)
                self.Trend_Linear = nn.Linear(self.seq_len, self.pred_len)
                self.time_Linear = nn.Linear(72*4, 72, bias=False)
            
            if self.version == 'TCC':
                self.Seasonal_Linear = nn.Linear(self.seq_len*2, self.pred_len)
                self.Trend_Linear = nn.Linear(self.seq_len*2, self.pred_len)
                self.time_Conv = nn.Conv1d(in_channels=4, out_channels=1, kernel_size=1, bias=False)
            
            if self.version == 'TCS':
                self.Seasonal_Linear = nn.Linear(self.seq_len, self.pred_len)
                self.Trend_Linear = nn.Linear(self.seq_len, self.pred_len)
                self.time_Conv = nn.Conv1d(in_channels=4, out_channels=1, kernel_size=1, bias=False)
                
            if self.version == 'TCCC':
                self.Seasonal_power_Linear = nn.Linear(self.seq_len, self.seq_len)
                self.Trend_power_Linear = nn.Linear(self.seq_len, self.seq_len)
                self.Seasonal_Linear = nn.Linear(self.seq_len, self.pred_len)
                self.Trend_Linear = nn.Linear(self.seq_len, self.pred_len)
                self.Seasonal_power_time_Conv = nn.Conv1d(in_channels=2, out_channels=1, kernel_size=1)
                # self.Trend_power_time_Conv = nn.Conv1d(in_channels=2, out_channels=1, kernel_size=1)
                self.time_Conv = nn.Conv1d(in_channels=4, out_channels=1, kernel_size=1, bias=False)
            
            if self.version == 'TPE' or 'TPS':
                self.Seasonal_Linear = nn.Linear(self.seq_len, self.pred_len)
                self.Trend_Linear = nn.Linear(self.seq_len, self.pred_len)
                self.time_Conv = nn.Conv1d(in_channels=4, out_channels=1, kernel_size=1, bias=False)
                self.Seasonal_power_Linear = nn.Linear(self.seq_len, self.seq_len)
                self.Trend_power_Linear = nn.Linear(self.seq_len, self.seq_len)
                self.time_linear = nn.Linear(self.seq_len, self.seq_len)
                
            if self.version == 'TCM':
                self.Seasonal_Linear = nn.Linear(self.seq_len, self.pred_len)
                self.Trend_Linear = nn.Linear(self.seq_len, self.pred_len)
                self.time_Conv = nn.Conv1d(in_channels=4, out_channels=1, kernel_size=1, bias=False)
            
            if self.version == 'TWM':
                self.Seasonal_Linear = nn.Linear(self.seq_len, self.pred_len)
                self.Trend_Linear = nn.Linear(self.seq_len, self.pred_len)
                self.time_Conv = nn.Conv1d(in_channels=4, out_channels=1, kernel_size=1, bias=False)
                # self.sigmoid = nn.Sigmoid()
                # self.time_Linear = nn.Linear(self.seq_len, self.seq_len)
            
            if self.version == 'TWM-2':
                self.Seasonal_Linear = nn.Linear(self.seq_len, self.pred_len)
                self.Trend_Linear = nn.Linear(self.seq_len, self.pred_len)
                self.time_Conv = nn.Conv1d(in_channels=4, out_channels=1, kernel_size=1, bias=False)
                self.power_Seasonal_Linear = nn.Linear(self.seq_len, self.seq_len)
                self.power_Trend_Linear = nn.Linear(self.seq_len, self.seq_len)                
                # self.sigmoid = nn.Sigmoid()
                # self.time_Linear = nn.Linear(self.seq_len, self.seq_len)
                
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
            
            if self.version == 'TLC':
                x_mark = x_mark.reshape(-1, 72*4)
                time = self.time_Linear(x_mark).unsqueeze(2)
                seasonal_init = torch.cat([seasonal_init, time], dim=1)
                trend_init = torch.cat([trend_init, time], dim=1)
                seasonal_output = self.Seasonal_Linear(seasonal_init.permute(0,2,1)).permute(0,2,1)
                trend_output = self.Trend_Linear(trend_init.permute(0,2,1)).permute(0,2,1)
                
            if self.version == 'TLS':
                x_mark = x_mark.reshape(-1, 72*4)
                time = self.time_Linear(x_mark).unsqueeze(2)
                seasonal_init = seasonal_init + time
                trend_init = trend_init + time
                seasonal_output = self.Seasonal_Linear(seasonal_init.permute(0,2,1)).permute(0,2,1)
                trend_output = self.Trend_Linear(trend_init.permute(0,2,1)).permute(0,2,1)
            
            if self.version == 'TCC':
                time = self.time_Conv(x_mark.permute(0,2,1)).permute(0,2,1)
                seasonal_init = torch.cat([seasonal_init, time], dim=1)
                trend_init = torch.cat([trend_init, time], dim=1)
                seasonal_output = self.Seasonal_Linear(seasonal_init.permute(0,2,1)).permute(0,2,1)
                trend_output = self.Trend_Linear(trend_init.permute(0,2,1)).permute(0,2,1)
            
            if self.version == 'TCS':
                time = self.time_Conv(x_mark.permute(0,2,1)).permute(0,2,1)
                seasonal_init = seasonal_init + time
                trend_init = trend_init + time
                seasonal_output = self.Seasonal_Linear(seasonal_init.permute(0,2,1)).permute(0,2,1)
                trend_output = self.Trend_Linear(trend_init.permute(0,2,1)).permute(0,2,1)
                    
            if self.version == 'TCCC':
                seasonal_init = self.Seasonal_power_Linear(seasonal_init.permute(0,2,1)).permute(0,2,1)
                trend_init = self.Trend_power_Linear(trend_init.permute(0,2,1)).permute(0,2,1)
                time = self.time_Conv(x_mark.permute(0,2,1)).permute(0,2,1)
                seasonal_init = torch.cat([seasonal_init, time], dim=2)
                # trend_init = torch.cat([trend_init, time], dim=2)
                seasonal_init = self.Seasonal_power_time_Conv(seasonal_init.permute(0,2,1))
                # trend_init = self.Trend_power_time_Conv(trend_init.permute(0,2,1))
                seasonal_output = self.Seasonal_Linear(seasonal_init).permute(0,2,1)
                trend_output = self.Trend_Linear(trend_init.permute(0,2,1)).permute(0,2,1)
            
            if self.version == 'TPS':
                time = self.time_Conv(x_mark.permute(0,2,1)).permute(0,2,1)
                seasonal_power = self.Seasonal_power_Linear(seasonal_init.permute(0,2,1)).permute(0,2,1)
                # trend_power = self.Trend_power_Linear(trend_init.permute(0,2,1)).permute(0,2,1)
                seasonal_init = seasonal_power + time
                # trend_init = trend_power + time
                seasonal_output = self.Seasonal_Linear(seasonal_init.permute(0,2,1)).permute(0,2,1)
                trend_output = self.Trend_Linear(trend_init.permute(0,2,1)).permute(0,2,1)
                            
            if self.version == 'TCM':
                time = self.time_Conv(x_mark.permute(0,2,1)).permute(0,2,1)
                seasonal_init = seasonal_init * time
                trend_init = trend_init * time
                seasonal_output = self.Seasonal_Linear(seasonal_init.permute(0,2,1)).permute(0,2,1)
                trend_output = self.Trend_Linear(trend_init.permute(0,2,1)).permute(0,2,1)        
                
            if self.version == 'TPM':
                time = self.time_Conv(x_mark.permute(0,2,1)).permute(0,2,1)
                seasonal_power = self.Seasonal_power_Linear(seasonal_init.permute(0,2,1)).permute(0,2,1)
                trend_power = self.Trend_power_Linear(trend_init.permute(0,2,1)).permute(0,2,1)
                seasonal_init = seasonal_power * time
                trend_init = trend_power * time
                seasonal_output = self.Seasonal_Linear(seasonal_init.permute(0,2,1)).permute(0,2,1)
                trend_output = self.Trend_Linear(trend_init.permute(0,2,1)).permute(0,2,1)
            
            if self.version == 'TWM':
                time = self.time_Conv(x_mark.permute(0,2,1)).permute(0,2,1)
                time = F.softmax(time, dim=1) 
                # time = self.sigmoid(time)
                # time = self.time_Linear(time.permute(0,2,1)).permute(0,2,1)
                seasonal_init = seasonal_init * time
                trend_init = trend_init * time
                seasonal_output = self.Seasonal_Linear(seasonal_init.permute(0,2,1)).permute(0,2,1)
                trend_output = self.Trend_Linear(trend_init.permute(0,2,1)).permute(0,2,1)

            if self.version == 'TWM-2':
                time = self.time_Conv(x_mark.permute(0,2,1)).permute(0,2,1)
                seasonal_init = self.Seasonal_power_Linear(seasonal_init.permute(0,2,1)).permute(0,2,1)
                trend_init = self.Trend_power_Linear(trend_init.permute(0,2,1)).permute(0,2,1)
                time = F.softmax(time, dim=1)
                # time = self.sigmoid(time)
                # time = self.time_Linear(time.permute(0,2,1)).permute(0,2,1)
                seasonal_init = seasonal_init * time
                trend_init = trend_init * time
                seasonal_output = self.Seasonal_Linear(seasonal_init.permute(0,2,1)).permute(0,2,1)
                trend_output = self.Trend_Linear(trend_init.permute(0,2,1)).permute(0,2,1)
                
            if self.version == 'D':
                seasonal_init = self.Seasonal_Linear1(seasonal_init.permute(0,2,1)).permute(0,2,1)
                trend_init = self.Trend_Linear1(trend_init.permute(0,2,1)).permute(0,2,1)
                seasonal_init = self.relu(seasonal_init)
                trend_init = self.relu(trend_init)
                seasonal_output = self.Seasonal_Linear2(seasonal_init.permute(0,2,1)).permute(0,2,1)
                trend_output = self.Trend_Linear2(trend_init.permute(0,2,1)).permute(0,2,1)

        x = seasonal_output + trend_output
        return x # to [Batch, Output length, Channel]
