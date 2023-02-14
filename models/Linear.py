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
        
        if self.version == 'B':
            self.Linear = nn.Linear(self.seq_len, self.pred_len)
        
        elif self.version == 'TLC':
            self.Linear = nn.Linear(self.seq_len*2, self.pred_len)
            self.time_Linear = nn.Linear(72*4, 72, bias=False)
        
        elif self.version == 'D':
            self.Linear1 = nn.Linear(self.seq_len, 336)
            self.Linear2 = nn.Linear(336, self.pred_len)
            self.relu = nn.ReLU()

        elif self.version == 'TLS':
            self.Linear = nn.Linear(self.seq_len, self.pred_len)
            self.time_Linear = nn.Linear(72*4, 72, bias=False)
        
        elif self.version == 'TCC':
            self.Linear = nn.Linear(self.seq_len*2, self.pred_len)
            self.time_Conv = nn.Conv1d(in_channels=4, out_channels=1, kernel_size=1, bias=False)
        
        elif self.version == 'TCS':
            self.Linear = nn.Linear(self.seq_len, self.pred_len)
            self.time_Conv = nn.Conv1d(in_channels=4, out_channels=1, kernel_size=1, bias=False)
            
        elif self.version == 'TCCC':
            self.Linear = nn.Linear(self.seq_len, self.pred_len)
            self.power_time_Conv = nn.Conv1d(in_channels=2, out_channels=1, kernel_size=1)
            self.time_Conv = nn.Conv1d(in_channels=4, out_channels=1, kernel_size=1, bias=False)
        
        elif self.version == 'TPE' or 'TPS':
            self.Linear = nn.Linear(self.seq_len, self.pred_len)
            self.time_Conv = nn.Conv1d(in_channels=4, out_channels=1, kernel_size=1, bias=False)
            self.power_Linear = nn.Linear(self.seq_len, self.seq_len)

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
            
            elif self.version == 'TLC':
                x_mark = x_mark.reshape(-1, 72*4)
                time = self.time_Linear(x_mark).unsqueeze(2)
                x = torch.cat([x, time], dim=1)
                x = self.Linear(x.permute(0,2,1)).permute(0,2,1)
                
            elif self.version == 'TLS':
                x_mark = x_mark.reshape(-1, 72*4)
                time = self.time_Linear(x_mark).unsqueeze(2)
                x = x+time
                x = self.Linear(x.permute(0,2,1)).permute(0,2,1)
            
            elif self.version == 'TCC':
                time = self.time_Conv(x_mark.permute(0,2,1)).permute(0,2,1)
                x = torch.cat([x, time], dim=1)
                x = self.Linear(x.permute(0,2,1)).permute(0,2,1)
            
            elif self.version == 'TCS':
                time = self.time_Conv(x_mark.permute(0,2,1)).permute(0,2,1)
                x = x+time
                x = self.Linear(x.permute(0,2,1)).permute(0,2,1)
                    
            elif self.version == 'TCCC':
                time = self.time_Conv(x_mark.permute(0,2,1)).permute(0,2,1)
                x = torch.cat([x, time], dim=2)
                x = self.power_time_Conv(x.permute(0,2,1))
                x = self.Linear(x).permute(0,2,1)
            
            elif self.version == 'TPE':
                time = self.time_Conv(x_mark.permute(0,2,1)).permute(0,2,1)
                power = self.power_Linear(x.permute(0,2,1)).permute(0,2,1)
                x = power * time
                x = self.Linear(x.permute(0,2,1)).permute(0,2,1)
            
            elif self.version == 'TPS':
                time = self.time_Conv(x_mark.permute(0,2,1)).permute(0,2,1)
                power = self.power_Linear(x.permute(0,2,1)).permute(0,2,1)
                x = power+time
                x = self.Linear(x.permute(0,2,1)).permute(0,2,1)
            
            else:
                x = self.Linear1(x.permute(0,2,1)).permute(0,2,1)
                x = self.relu(x)
                x = self.Linear2(x.permute(0,2,1)).permute(0,2,1)
                        
        return x # [Batch, Output length, Channel]