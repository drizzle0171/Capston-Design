import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Model(nn.Module):
    """
    Normalization-Linear
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.channels = configs.enc_in
        self.individual = configs.individual
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
            self.Linear = nn.ModuleList()
            for i in range(self.channels):
                self.Linear.append(nn.Linear(self.seq_len,self.pred_len))

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
            self.Linear = nn.Linear(self.seq_len, self.pred_len)
            self.sigmoid = nn.Sigmoid()
            self.time_Conv = nn.Conv1d(in_channels=self.time_channel, out_channels=1, kernel_size=1, bias=False) # time channel 22         
            self.time_Linear = nn.Linear(self.seq_len, self.seq_len)

        if self.version == 'oneHot':
            self.Linear = nn.Linear(self.seq_len, self.pred_len)
            self.power_Linear = nn.Linear(self.seq_len, self.seq_len)
            self.sigmoid = nn.Sigmoid()
            self.time_Conv = nn.Conv1d(in_channels=self.time_channel, out_channels=1, kernel_size=1, bias=False) # time channel 22         
            self.time_Linear = nn.Linear(self.seq_len, self.seq_len)
               
        if self.version == 'oneHot-ReLU':
            self.Linear = nn.Linear(self.seq_len, self.pred_len)
            self.power_Linear = nn.Linear(self.seq_len, self.seq_len)
            self.sigmoid = nn.Sigmoid()
            self.relu = nn.ReLU()
            self.time_Conv = nn.Conv1d(in_channels=self.time_channel, out_channels=1, kernel_size=1, bias=False) # time channel 22         
            self.time_Linear = nn.Linear(self.seq_len, self.seq_len)
            
        if self.version == 'B':
            self.Linear = nn.Linear(self.seq_len, self.pred_len)

        if self.version == 'D':
            self.Linear1 = nn.Linear(self.seq_len, self.seq_len)
            self.Linear2 = nn.Linear(self.seq_len, self.pred_len)

            
        if self.version == 'D-ReLU':
            self.Linear1 = nn.Linear(self.seq_len, 72)
            self.Linear2 = nn.Linear(72, self.pred_len)
            self.relu = nn.ReLU()
        
        if self.version == 'B-TWM':
            self.Linear = nn.Linear(self.seq_len, self.pred_len)
            self.sigmoid = nn.Sigmoid()
            self.time_Conv = nn.Conv1d(in_channels=self.time_channel, out_channels=1, kernel_size=1, bias=False)            
            self.time_Linear = nn.Linear(self.seq_len, self.seq_len)
            
        if self.version == 'TWM':
            self.Linear = nn.Linear(self.seq_len, self.pred_len)
            self.power_Linear = nn.Linear(self.seq_len, self.seq_len)
            self.sigmoid = nn.Sigmoid()
            self.time_Conv = nn.Conv1d(in_channels=self.time_channel, out_channels=1, kernel_size=1, bias=False)            
            self.time_Linear = nn.Linear(self.seq_len, self.seq_len)
                             
        if self.version == 'TWM-ReLU':
            self.Linear = nn.Linear(self.seq_len, self.pred_len)
            self.power_Linear = nn.Linear(self.seq_len, self.seq_len)
            self.sigmoid = nn.Sigmoid()
            self.relu = nn.ReLU()
            self.time_Conv = nn.Conv1d(in_channels=self.time_channel, out_channels=1, kernel_size=1, bias=False)            
            self.time_Linear = nn.Linear(self.seq_len, self.seq_len)
 
    def forward(self, x, x_mark):
        # x: [Batch, Input length, Channel]
        seq_last = x[:,-1:,:].detach()
        x = x - seq_last
        
        if self.individual:
            output = torch.zeros([x.size(0),self.pred_len,x.size(2)],dtype=x.dtype).to(x.device)
            for i in range(self.channels):
                output[:,:,i] = self.Linear[i](x[:,:,i])
            x = output
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
                x = x*time 
                x = self.Linear(x.permute(0,2,1)).permute(0,2,1)

            if self.version == 'oneHot':
                x_mark = x[:,:,1:] # 전력 제외 one hot vector
                time = self.time_Conv(x_mark.permute(0,2,1)).permute(0,2,1)
                time = self.time_Linear(time.permute(0,2,1)).permute(0,2,1)
                time = self.sigmoid(time)
                x = self.power_Linear(x.permute(0,2,1)).permute(0,2,1)
                x = x*time 
                x = self.Linear(x.permute(0,2,1)).permute(0,2,1)

            if self.version == 'oneHot-ReLU':
                x_mark = x[:,:,1:] # 전력 제외 one hot vector
                time = self.time_Conv(x_mark.permute(0,2,1)).permute(0,2,1)
                time = self.time_Linear(time.permute(0,2,1)).permute(0,2,1)
                time = self.sigmoid(time)
                x = self.power_Linear(x.permute(0,2,1)).permute(0,2,1)
                x = self.relu(x)
                x = x*time
                x = self.Linear(x.permute(0,2,1)).permute(0,2,1)

            if self.version == 'B':
                x = self.Linear(x.permute(0,2,1)).permute(0,2,1)
            
            if self.version =='D':
                x = self.Linear1(x.permute(0,2,1)).permute(0,2,1)
                x = self.Linear2(x.permute(0,2,1)).permute(0,2,1)   
            
            if self.version =='D-ReLU':
                x = self.Linear1(x.permute(0,2,1)).permute(0,2,1)
                x = self.relu(x)
                x = self.Linear2(x.permute(0,2,1)).permute(0,2,1) 
            
            if self.version =='B-TWM':
                time = self.time_Conv(x_mark.permute(0,2,1)).permute(0,2,1)
                time = self.time_Linear(time.permute(0,2,1)).permute(0,2,1)
                time = self.sigmoid(time)
                x = x*time
                x = self.Linear(x.permute(0,2,1)).permute(0,2,1)
                
            if self.version == 'TWM':
                time = self.time_Conv(x_mark.permute(0,2,1)).permute(0,2,1)
                time = self.time_Linear(time.permute(0,2,1)).permute(0,2,1)
                time = self.sigmoid(time)
                x = self.power_Linear(x.permute(0,2,1)).permute(0,2,1)
                x = x*time 
                x = self.Linear(x.permute(0,2,1)).permute(0,2,1)

            if self.version == 'TWM-ReLU':
                time = self.time_Conv(x_mark.permute(0,2,1)).permute(0,2,1)
                time = self.time_Linear(time.permute(0,2,1)).permute(0,2,1)
                time = self.sigmoid(time)
                x = self.power_Linear(x.permute(0,2,1)).permute(0,2,1)
                x = self.relu(x)
                x = x*time 
                x = self.Linear(x.permute(0,2,1)).permute(0,2,1)    
            
        x = x + seq_last
        return x # [Batch, Output length, Channel]