import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Linear(nn.Module):
    """
    Just one Linear layer
    """
    def __init__(self, seq_len, pred_len):
        super(Linear, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.Linear = nn.Linear(self.seq_len, self.pred_len)
   
    def forward(self, x, x_mark):
        # x: [Batch, Input length, Channel]
        x = self.Linear(x.permute(0,2,1)).permute(0,2,1)     
        return x # [Batch, Output length, Channel]

class betterLinear(nn.Module):
    def __init__(self, seq_len, pred_len):
        super(betterLinear, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.Linear = nn.Linear(self.seq_len, self.pred_len)
        self.power_Linear = nn.Linear(self.seq_len, self.seq_len)
        self.sigmoid = nn.Sigmoid()
        self.time_Conv = nn.Conv1d(in_channels=4, out_channels=1, kernel_size=1, bias=False)            
        self.time_Linear = nn.Linear(self.seq_len, self.seq_len)

    def forward(self, x, x_mark):
        time = self.time_Conv(x_mark.permute(0,2,1)).permute(0,2,1)
        time = self.time_Linear(time.permute(0,2,1)).permute(0,2,1)
        time = self.sigmoid(time)
        power = self.power_Linear(x.permute(0,2,1)).permute(0,2,1)
        x = power*time
        x = self.Linear(x.permute(0,2,1)).permute(0,2,1)
        return x