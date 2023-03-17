import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Model(nn.Module):
    """
    Just one Linear layer
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
        
        if self.version == 'B':
            self.Linear = nn.Linear(self.seq_len, self.pred_len)

        if self.version == 'D':
            self.Linear1 = nn.Linear(self.seq_len, 72)
            self.Linear2 = nn.Linear(72, self.pred_len)
            self.sigmoid = nn.ReLU()
                                   
        if self.version == 'TWM-2':
            self.Linear = nn.Linear(self.seq_len, self.pred_len)
            self.power_Linear = nn.Linear(self.seq_len, self.seq_len)
            self.sigmoid = nn.Sigmoid()
            self.relu = nn.ReLU()
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

        if self.individual:
            self.Linear = nn.ModuleList()
            for i in range(self.channels):
                self.Linear.append(nn.Linear(self.seq_len,self.pred_len))


    def forward(self, x, x_mark):
        # x: [Batch, Input length, Channel]
        if self.individual:
            output = torch.zeros([x.size(0),self.pred_len,x.size(2)],dtype=x.dtype).to(x.device)
            for i in range(self.channels):
                output[:,:,i] = self.Linear[i](x[:,:,i])
            x = output
        else:
            
            if self.version == 'B':
                x = self.Linear(x.permute(0,2,1)).permute(0,2,1)

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
                power = self.power_Linear(x.permute(0,2,1)).permute(0,2,1)
                x = power*time 
                x = self.Linear(x.permute(0,2,1)).permute(0,2,1)

            if self.version == 'D':
                x = self.Linear1(x.permute(0,2,1)).permute(0,2,1)
                x = self.sigmoid(x)
                x = self.Linear2(x.permute(0,2,1)).permute(0,2,1)      

        return x # [Batch, Output length, Channel]