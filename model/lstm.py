'''
Author: FreeeBird
Date: 2022-04-13 20:16:14
LastEditTime: 2022-06-10 16:31:06
LastEditors: FreeeBird
Description: 
FilePath: /TM_Prediction_With_Missing_Data/model/lstm.py
'''
import torch.nn as nn
import torch




class LSTM(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_layer=3,dropout=0.2):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(
            in_dim, hidden_dim, n_layer, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, in_dim)

    def forward(self, x, mask=None):
        
        if mask is not None:
            x = x * (1-mask)
        x = x.squeeze()
        # print(x.size())
        x, _ = self.lstm(x)
        x = self.fc(x[:, -1, :])  
        return x
    
    


class LSTM2D(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_layer=1,dropout=0.2):
        super(LSTM2D, self).__init__()
        self.t_lstm = nn.LSTM(1, hidden_dim, 1, batch_first=True,bias=True,bidirectional=False)
        self.f_lstm = nn.LSTM(1, hidden_dim, 1, batch_first=True,bias=True,bidirectional=False)
        self.fc = nn.Linear(2*hidden_dim, 1)

    def forward(self, x, mask=None):
        if mask is not None:
            x = x * (1-mask)
        BS,T,F = x.size()
        x = x.unsqueeze(-1) # BS,T,F,1
        f,_ = self.f_lstm(x.reshape(-1,F,1))
        f = f.reshape([BS,T,F,-1])
        t, _ = self.t_lstm(x.permute(0,2,1,3).reshape(-1,T,1))
        t = t.reshape([BS,F,T,-1]).permute(0,2,1,3)
        x = torch.cat([t,f],dim=-1) # BS,T,F,D
        x = self.fc(x[:, -1, :]).squeeze()
        return x

class LSTM3D(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_layer=1,dropout=0.2):
        super(LSTM3D, self).__init__()
        self.t_lstm = nn.LSTM(1, hidden_dim, 1, batch_first=True,bias=True,bidirectional=True)
        self.f_lstm = nn.LSTM(1, hidden_dim, 1, batch_first=True,bias=True,bidirectional=True)
        self.f_lstm = nn.LSTM(1, hidden_dim, 1, batch_first=True,bias=True,bidirectional=True)
        self.fc = nn.Linear(6*hidden_dim, 1)

    def forward(self, x, mask=None):
        if mask is not None:
            x = x * (1-mask)
        BS,T,F = x.size()
        x = x.unsqueeze(-1) # BS,T,F,1
        f,_ = self.f_lstm(x.reshape(-1,F,1))
        f = f.reshape([BS,T,F,-1])
        t, _ = self.t_lstm(x.permute(0,2,1,3).reshape(-1,T,1))
        t = t.reshape([BS,F,T,-1]).permute(0,2,1,3)
        x = torch.cat([t,f],dim=-1) # BS,T,F,D
        x = self.fc(x[:, -1, :]).squeeze()
        return x
    