'''
Author: FreeeBird
Date: 2022-04-18 20:58:39
LastEditTime: 2022-10-28 15:05:40
LastEditors: FreeeBird
Description: 
FilePath: /TM_Prediction_With_Missing_Data/model/unet.py
'''
""" Parts of the U-Net model """

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None,kernel_size=3):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=kernel_size, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=kernel_size, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels,mp=2,kernel_size=3):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(mp),
            DoubleConv(in_channels, out_channels,kernel_size)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 32)
        self.down1 = Down(32, 64,2)
        self.down2 = Down(64, 128,2)
        self.down3 = Down(128, 256,2)
        factor = 2 if bilinear else 1
        self.down4 = Down(256, 512 // factor)

        self.up1 = Up(512, 256 // factor, bilinear)
        self.up2 = Up(256, 128 // factor, bilinear)
        self.up3 = Up(128, 64 // factor, bilinear)
        self.up4 = Up(64, 32, bilinear)
        self.outc = OutConv(32, n_classes)
        # self.act = torch.nn.Sigmoid()

    def forward(self, x,mask=None):
        if mask is not None:
            x = x * (1-mask) 
        if len(x.size())<4:
            BS,T,F = x.size()
            x = x.unsqueeze(1) # BS,1,T,F
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        # print(x5.size())
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        # x = self.act(x)
        return x
    
class UNet_BKS(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False,kernel_size=7):
        super(UNet_BKS, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 32)
        self.down1 = Down(32, 64,2,kernel_size)
        self.down2 = Down(64, 128,2,kernel_size)
        self.down3 = Down(128, 256,2,kernel_size)
        factor = 2 if bilinear else 1
        self.down4 = Down(256, 512 // factor)
        self.up1 = Up(512, 256 // factor, bilinear)
        self.up2 = Up(256, 128 // factor, bilinear)
        self.up3 = Up(128, 64 // factor, bilinear)
        self.up4 = Up(64, 32, bilinear)
        self.outc = OutConv(32, n_classes)
        # self.act = torch.nn.Sigmoid()

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        # print(x5.size())
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        # x = self.act(x)
        return x
    
    


# class STUNet(nn.Module):
#     def __init__(self, n_channels, n_classes, bilinear=False,in_dim=144, hidden_dim=144, n_layer=3,dropout=0.2):
#         super(STUNet, self).__init__()
#         self.n_channels = n_channels
#         self.n_classes = n_classes
#         self.bilinear = bilinear
#         self.inc = DoubleConv(n_channels, 32)
#         self.down1 = Down(32, 64)
#         self.down2 = Down(64, 128)
#         self.down3 = Down(128, 256)
#         factor = 2 if bilinear else 1
#         self.down4 = Down(256, 512 // factor,1)
#         self.bilstm = nn.LSTM(512,256,1,False,batch_first=True,bidirectional=True)
#         self.up1 = Up(512, 256 // factor, bilinear)
#         self.up2 = Up(256, 128 // factor, bilinear)
#         self.up3 = Up(128, 64 // factor, bilinear)
#         self.up4 = Up(64, 32, bilinear)
#         self.outc = OutConv(32, n_classes)
#         self.lstm = nn.LSTM(
#             in_dim, hidden_dim, n_layer, batch_first=True,bidirectional=False, dropout=dropout)
#         self.fc = nn.Linear(hidden_dim, in_dim)
#         self.node = int(math.sqrt(in_dim))

#     def forward(self, x, mask=None):
#         if mask is not None:
#             x = x * (1-mask) 
#         BS,T,F = x.size()
#         x_hat = x.unsqueeze(2).reshape([BS*T,1,self.node,self.node]) # BST,1,N,N
#         x1 = self.inc(x_hat)
#         x2 = self.down1(x1)
#         x3 = self.down2(x2)
#         x4 = self.down3(x3)
#         x5 = self.down4(x4) # BS
#         x5 = x5.reshape([BS,T,-1]) # BS,T,512
#         x5,_ = self.bilstm(x5) # BS,T,512
#         x5 = x5.reshape([BS*T,512,1,1])
#         x_hat = self.up1(x5, x4)
#         x_hat = self.up2(x_hat, x3)
#         x_hat = self.up3(x_hat, x2)
#         x_hat = self.up4(x_hat, x1)
#         x_hat = self.outc(x_hat)
#         x_hat = x_hat.reshape([BS,T,F])
#         # x_hat = x_hat * mask + x * (1-mask)
#         x, _ = self.lstm(x_hat)
#         x = self.fc(x[:, -1, :])  
#         return x,x_hat

class UNet_LSTM(nn.Module):
    def __init__(self, n_channels=2, bilinear=True,in_dim=144, hidden_dim=144, n_layer=3,dropout=0.2):
        super(UNet_LSTM, self).__init__()
        self.in_chan = n_channels
        self.unet = UNet(n_channels,1,bilinear)
        self.lstm = nn.LSTM(
            in_dim, hidden_dim, n_layer, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, in_dim)
        # self.node = int(math.sqrt(in_dim))
    def forward(self, x, mask=None):
        if mask is not None:
            x = x * (1-mask) 
        BS,T,F = x.size()
        if self.in_chan == 1:
            x_hat = x.unsqueeze(1) # BS,1,T,F
        else:
            x_hat = torch.stack([x,mask],dim=1)
        # x_hat = x.reshape(BS,T,self.node,self.node)
        x_hat = self.unet(x_hat)
        x_hat = x_hat.reshape([BS,T,F])
        # x_hat = x_hat * mask + x * (1-mask)
        # x, _ = self.lstm(x_hat)
        x, _ = self.lstm(x_hat * mask + x * (1-mask))
        x = self.fc(x[:, -1, :])  
        return x,x_hat


class BKSUNet_LSTM(nn.Module):
    def __init__(self, n_channels=2, bilinear=True,in_dim=144, hidden_dim=144, n_layer=3,dropout=0.2,kernel_size=7):
        super(BKSUNet_LSTM, self).__init__()
        self.in_chan = n_channels
        self.unet = UNet_BKS(n_channels,1,bilinear,kernel_size)
        self.lstm = nn.LSTM(
            in_dim, hidden_dim, n_layer, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, in_dim)
        # self.node = int(math.sqrt(in_dim))
    def forward(self, x, mask=None):
        if mask is not None:
            x = x * (1-mask) 
        BS,T,F = x.size()
        if self.in_chan == 1:
            x_hat = x.unsqueeze(1) # BS,1,T,F
        else:
            x_hat = torch.stack([x,mask],dim=1)
        # x_hat = x.reshape(BS,T,self.node,self.node)
        x_hat = self.unet(x_hat)
        x_hat = x_hat.reshape([BS,T,F])
        # x_hat = x_hat * mask + x * (1-mask)
        # x, _ = self.lstm(x_hat)
        x, _ = self.lstm(x_hat * mask + x * (1-mask))
        x = self.fc(x[:, -1, :])  
        return x,x_hat


# 
class UNet_LSTM2D(nn.Module):
    def __init__(self, n_channels=2, bilinear=True,in_dim=144, hidden_dim=144, n_layer=3,dropout=0.2):
        super(UNet_LSTM2D, self).__init__()
        self.in_chan = n_channels
        self.unet = UNet(n_channels,1,bilinear)
        self.t_lstm = nn.LSTM(1, hidden_dim, 1, batch_first=True,bias=True,bidirectional=False)
        self.f_lstm = nn.LSTM(1, hidden_dim, 1, batch_first=True,bias=True,bidirectional=False)
        self.fc = nn.Linear(2*hidden_dim, 1)
        # self.node = int(math.sqrt(in_dim))
    def forward(self, x, mask=None):
        if mask is not None:
            x = x * (1-mask) 
        BS,T,F = x.size()
        if self.in_chan == 1:
            x_hat = x.unsqueeze(1) # BS,1,T,F
        else:
            x_hat = torch.stack([x,mask],dim=1)
        # x_hat = x.reshape(BS,T,self.node,self.node)
        x_hat = self.unet(x_hat)
        x_hat = x_hat.reshape([BS,T,F])
        # x_hat = x_hat * mask + x * (1-mask)
        # x, _ = self.lstm(x_hat)
        x = x_hat * mask + x * (1-mask)
        x = x.unsqueeze(-1) # BS,T,F,1
        f,_ = self.f_lstm(x.reshape(-1,F,1))
        f = f.reshape([BS,T,F,-1])
        t, _ = self.t_lstm(x.permute(0,2,1,3).reshape(-1,T,1))
        t = t.reshape([BS,F,T,-1]).permute(0,2,1,3)
        x = torch.cat([t,f],dim=-1) # BS,T,F,D
        x = self.fc(x[:, -1, :]).squeeze()
        return x,x_hat
    
    
if __name__ == '__main__':
    model = UNet(1,1,True)
    x = torch.zeros([32,1,26,144])
    x = model(x)
    print(x.size())
    x = torch.zeros([32,1,26,529])
    x = model(x)
    print(x.size())
    pass
