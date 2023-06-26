'''
Author: FreeeBird
Date: 2022-04-19 21:18:11
LastEditTime: 2022-10-28 15:34:16
LastEditors: FreeeBird
Description: 
FilePath: /TM_Prediction_With_Missing_Data/model/unet3d.py
'''
import math
from tkinter import Y
import torch 
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, mid_channels, kernel_size=3,stride=1, padding=1, bias=False),
            nn.BatchNorm3d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(mid_channels, out_channels, kernel_size=3,stride=1, padding=1,bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels,mp_straide=2):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool3d(mp_straide),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=False):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose3d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        # CDHW in 3D
        # print(x1.size())
        # print(x2.size())
        diffZ = x2.size()[2] - x1.size()[2]
        diffY = x2.size()[3] - x1.size()[3]
        diffX = x2.size()[4] - x1.size()[4]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2,
                        diffZ // 2, diffZ - diffZ // 2,])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)
    

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

# other implement
class UNet3D(nn.Module):
    def __init__(self, in_channel, n_classes):
        self.in_channel = in_channel
        self.n_classes = n_classes
        super(UNet3D, self).__init__()
        self.ec0 = self.encoder(self.in_channel, 32, bias=False, batchnorm=False)
        self.ec1 = self.encoder(32, 64, bias=False, batchnorm=False)
        self.ec2 = self.encoder(64, 64, bias=False, batchnorm=False)
        self.ec3 = self.encoder(64, 128, bias=False, batchnorm=False)
        self.ec4 = self.encoder(128, 128, bias=False, batchnorm=False)
        self.ec5 = self.encoder(128, 256, bias=False, batchnorm=False)
        self.ec6 = self.encoder(256, 256, bias=False, batchnorm=False)
        self.ec7 = self.encoder(256, 512, bias=False, batchnorm=False)

        self.pool0 = nn.MaxPool3d(2)
        self.pool1 = nn.MaxPool3d(2)
        self.pool2 = nn.MaxPool3d(2)

        self.dc9 = self.decoder(512, 512, kernel_size=2, stride=2, bias=False)
        self.dc8 = self.decoder(256 + 512, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.dc7 = self.decoder(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.dc6 = self.decoder(256, 256, kernel_size=2, stride=2, bias=False)
        self.dc5 = self.decoder(128 + 256, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.dc4 = self.decoder(128, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.dc3 = self.decoder(128, 128, kernel_size=2, stride=2, bias=False)
        self.dc2 = self.decoder(64 + 128, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.dc1 = self.decoder(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.dc0 = self.decoder(64, n_classes, kernel_size=1, stride=1, bias=False)


    def encoder(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1,
                bias=True, batchnorm=False):
        if batchnorm:
            layer = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
                nn.BatchNorm3d(out_channels),
                nn.ReLU())
        else:
            layer = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
                nn.ReLU())
        return layer


    def decoder(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                output_padding=0, bias=True):
        layer = nn.Sequential(
            nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride,
                               padding=padding, output_padding=output_padding, bias=bias),
            nn.ReLU())
        return layer

    def forward(self, x):
        e0 = self.ec0(x)
        syn0 = self.ec1(e0)
        e1 = self.pool0(syn0)
        e2 = self.ec2(e1)
        syn1 = self.ec3(e2)
        del e0, e1, e2

        e3 = self.pool1(syn1)
        e4 = self.ec4(e3)
        syn2 = self.ec5(e4)
        del e3, e4

        e5 = self.pool2(syn2)
        e6 = self.ec6(e5)
        e7 = self.ec7(e6)
        del e5, e6

        d9 = self.dc9(e7)
        d9 = torch.cat((d9, syn2))
        del e7, syn2

        d8 = self.dc8(d9)
        d7 = self.dc7(d8)
        del d9, d8

        d6 = torch.cat((self.dc6(d7), syn1))
        del d7, syn1

        d5 = self.dc5(d6)
        d4 = self.dc4(d5)
        del d6, d5

        d3 = torch.cat((self.dc3(d4), syn0))
        del d4, syn0

        d2 = self.dc2(d3)
        d1 = self.dc1(d2)
        del d3, d2

        d0 = self.dc0(d1)
        return d0
    

# UNet_3D
class UNet_3D(nn.Module):
    def __init__(self,in_dim, n_channels=1, n_classes=1, bilinear=False):
        super(UNet_3D, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 32)
        self.down1 = Down(32, 64)
        self.down2 = Down(64, 128)
        self.down3 = Down(128, 256)
        factor = 2 if bilinear else 1
        self.down4 = Down(256, 512 // factor,1)
        self.up1 = Up(512, 256 // factor, bilinear)
        self.up2 = Up(256, 128 // factor, bilinear)
        self.up3 = Up(128, 64 // factor, bilinear)
        self.up4 = Up(64, 32, bilinear)
        self.outc = OutConv(32, n_classes)
        self.h = int(math.sqrt(in_dim))
        
    # def restruction(self,x_res,fx,bx):
    #     bx = bx.flip(1)
    #     bx = bx + x_res
    #     bx[1:] = bx[1:] + fx[:-1]
    #     return bx

    def forward(self, x, mask=None):
        # if mask is not None:
        #     x = x * (1-mask)
        BS,T,F = x.size()
        x1 = x.unsqueeze(1).reshape([BS,1,T,self.h,self.h])
        # x1 = x.reshape([BS,T,self.h,self.h])
        x1 = self.inc(x1)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        # print(x5.size())
        x_hat = self.up1(x5, x4)
        x_hat = self.up2(x_hat, x3)
        x_hat = self.up3(x_hat, x2)
        x_hat = self.up4(x_hat, x1)
        x_hat = self.outc(x_hat)
        del x1,x2,x3,x4,x5
        x_hat = x_hat.reshape([BS,T,F])
        x_res = x_hat * mask + x * (1-mask)
        return x_res
    

class UNet3D_LSTM(nn.Module):
    def __init__(self,in_dim, hidden_dim, n_layer=3,dropout=0.2, n_channels=1, n_classes=1, bilinear=False):
        super(UNet3D_LSTM, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 32)
        self.down1 = Down(32, 64)
        self.down2 = Down(64, 128)
        self.down3 = Down(128, 256)
        factor = 2 if bilinear else 1
        self.down4 = Down(256, 512 // factor,1)
        self.up1 = Up(512, 256 // factor, bilinear)
        self.up2 = Up(256, 128 // factor, bilinear)
        self.up3 = Up(128, 64 // factor, bilinear)
        self.up4 = Up(64, 32, bilinear)
        self.outc = OutConv(32, n_classes)
        self.flstm = nn.LSTM(
            in_dim, in_dim, n_layer, batch_first=True, dropout=dropout)
        self.blstm = nn.LSTM(
            in_dim, in_dim, n_layer, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(in_dim, in_dim)
        self.h = int(math.sqrt(in_dim))
        self.sigmoid = nn.Sigmoid()
        
    def restruction(self,x_res,fx,bx):
        bx = bx.flip(1)
        bx = bx + x_res
        bx[1:] = bx[1:] + fx[:-1]
        return bx

    def forward(self, x, mask=None):
        # src_x = x
        if mask is not None:
            x = x * (1-mask)
        BS,T,F = x.size()
        x1 = x.unsqueeze(1).reshape([BS,1,T,self.h,self.h])
        # x1 = x.reshape([BS,T,self.h,self.h])
        x1 = self.inc(x1)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        # print(x5.size())
        x_hat = self.up1(x5, x4)
        x_hat = self.up2(x_hat, x3)
        x_hat = self.up3(x_hat, x2)
        x_hat = self.up4(x_hat, x1)
        x_hat = self.outc(x_hat)
        del x1,x2,x3,x4,x5
        # print(x_hat.size())
        x_hat = x_hat.reshape([BS,T,F])
        # x_hat = x_hat * mask + src_x * (1-mask)
        x_res = x_hat * mask + x * (1-mask)
        x,_ = self.flstm(x_res)
        y = self.fc(x[:, -1, :])  
        x_hat,_ = self.blstm(x.flip(dims=[1]))
        x_hat = self.restruction(x_res,x,x_hat)
        return y,x_hat
    
    
class UNet3D_LSTM2D(nn.Module):
    def __init__(self,in_dim, hidden_dim, n_layer=3,dropout=0.2, n_channels=1, n_classes=1, bilinear=False):
        super(UNet3D_LSTM2D, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 32)
        self.down1 = Down(32, 64)
        self.down2 = Down(64, 128)
        self.down3 = Down(128, 256)
        factor = 2 if bilinear else 1
        self.down4 = Down(256, 512 // factor,1)
        self.up1 = Up(512, 256 // factor, bilinear)
        self.up2 = Up(256, 128 // factor, bilinear)
        self.up3 = Up(128, 64 // factor, bilinear)
        self.up4 = Up(64, 32, bilinear)
        self.outc = OutConv(32, n_classes)
        self.t_lstm = nn.LSTM(1, hidden_dim, 1, batch_first=True,bias=True,bidirectional=False)
        self.f_lstm = nn.LSTM(1, hidden_dim, 1, batch_first=True,bias=True,bidirectional=False)
        self.fc = nn.Linear(2*hidden_dim, 1)
        self.h = int(math.sqrt(in_dim))


    def restruction(self,x,mask=None):
        # if mask is not None:
        #     x = x * (1-mask)
            
        BS,T,F = x.size()
        x1 = x.unsqueeze(1).reshape([BS,1,T,self.h,self.h])
        x1 = self.inc(x1)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        # print(x5.size())
        x_hat = self.up1(x5, x4)
        x_hat = self.up2(x_hat, x3)
        x_hat = self.up3(x_hat, x2)
        x_hat = self.up4(x_hat, x1)
        x_hat = self.outc(x_hat)
        del x1,x2,x3,x4,x5
        # print(x_hat.size())
        x_hat = x_hat.reshape([BS,T,F])
        # x_hat = x_hat * mask + src_x * (1-mask)
        x = x_hat * mask + x * (1-mask)
        return x
         
    def forward(self, x, mask=None):
        # src_x = x
        # if mask is not None:
        #     x = x * (1-mask)
        BS,T,F = x.size()
        x1 = x.unsqueeze(1).reshape([BS,1,T,self.h,self.h])
        x1 = self.inc(x1)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        # print(x5.size())
        x_hat = self.up1(x5, x4)
        x_hat = self.up2(x_hat, x3)
        x_hat = self.up3(x_hat, x2)
        x_hat = self.up4(x_hat, x1)
        x_hat = self.outc(x_hat)
        del x1,x2,x3,x4,x5
        # print(x_hat.size())
        x_hat = x_hat.reshape([BS,T,F])
        # x_hat = x_hat * mask + src_x * (1-mask)
        x = x_hat * mask + x * (1-mask)
        x = x.unsqueeze(-1) # BS,T,F,1
        f,_ = self.f_lstm(x.reshape(-1,F,1))
        f = f.reshape([BS,T,F,-1])
        t, _ = self.t_lstm(x.permute(0,2,1,3).reshape(-1,T,1))
        t = t.reshape([BS,F,T,-1]).permute(0,2,1,3)
        x = torch.cat([t,f],dim=-1) # BS,T,F,D
        x = self.fc(x[:, -1, :]).squeeze()
        return x,x_hat
    


    
class UNet3D_LSTMF(nn.Module):
    def __init__(self,in_dim, hidden_dim, n_layer=3,dropout=0.2, n_channels=1, n_classes=1, bilinear=False):
        super(UNet3D_LSTMF, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 32)
        self.down1 = Down(32, 64)
        self.down2 = Down(64, 128)
        self.down3 = Down(128, 256)
        factor = 2 if bilinear else 1
        self.down4 = Down(256, 512 // factor,1)
        self.up1 = Up(512, 256 // factor, bilinear)
        self.up2 = Up(256, 128 // factor, bilinear)
        self.up3 = Up(128, 64 // factor, bilinear)
        self.up4 = Up(64, 32, bilinear)
        self.outc = OutConv(32, n_classes)
        # self.t_lstm = nn.LSTM(1, hidden_dim, 1, batch_first=True,bias=True,bidirectional=False)
        self.f_lstm = nn.LSTM(1, hidden_dim, 1, batch_first=True,bias=True,bidirectional=False)
        self.fc = nn.Linear(hidden_dim, 1)
        self.h = int(math.sqrt(in_dim))


    def restruction(self,x,mask=None):
        # if mask is not None:
        #     x = x * (1-mask)
            
        BS,T,F = x.size()
        x1 = x.unsqueeze(1).reshape([BS,1,T,self.h,self.h])
        x1 = self.inc(x1)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        # print(x5.size())
        x_hat = self.up1(x5, x4)
        x_hat = self.up2(x_hat, x3)
        x_hat = self.up3(x_hat, x2)
        x_hat = self.up4(x_hat, x1)
        x_hat = self.outc(x_hat)
        del x1,x2,x3,x4,x5
        # print(x_hat.size())
        x_hat = x_hat.reshape([BS,T,F])
        # x_hat = x_hat * mask + src_x * (1-mask)
        x = x_hat * mask + x * (1-mask)
        return x
         
    def forward(self, x, mask=None):
        # src_x = x
        # if mask is not None:
        #     x = x * (1-mask)
        BS,T,F = x.size()
        x1 = x.unsqueeze(1).reshape([BS,1,T,self.h,self.h])
        x1 = self.inc(x1)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        # print(x5.size())
        x_hat = self.up1(x5, x4)
        x_hat = self.up2(x_hat, x3)
        x_hat = self.up3(x_hat, x2)
        x_hat = self.up4(x_hat, x1)
        x_hat = self.outc(x_hat)
        del x1,x2,x3,x4,x5
        # print(x_hat.size())
        x_hat = x_hat.reshape([BS,T,F])
        # x_hat = x_hat * mask + src_x * (1-mask)
        x = x_hat * mask + x * (1-mask)
        x = x.unsqueeze(-1) # BS,T,F,1
        x,_ = self.f_lstm(x.reshape(-1,F,1))
        x = x.reshape([BS,T,F,-1])
        # t, _ = self.t_lstm(x.permute(0,2,1,3).reshape(-1,T,1))
        # t = t.reshape([BS,F,T,-1]).permute(0,2,1,3)
        # x = torch.cat([t,f],dim=-1) # BS,T,F,D
        x = self.fc(x[:, -1, :]).squeeze()
        return x,x_hat
    

class UNet3D_LSTMT(nn.Module):
    def __init__(self,in_dim, hidden_dim, n_layer=3,dropout=0.2, n_channels=1, n_classes=1, bilinear=False):
        super(UNet3D_LSTMT, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 32)
        self.down1 = Down(32, 64)
        self.down2 = Down(64, 128)
        self.down3 = Down(128, 256)
        factor = 2 if bilinear else 1
        self.down4 = Down(256, 512 // factor,1)
        self.up1 = Up(512, 256 // factor, bilinear)
        self.up2 = Up(256, 128 // factor, bilinear)
        self.up3 = Up(128, 64 // factor, bilinear)
        self.up4 = Up(64, 32, bilinear)
        self.outc = OutConv(32, n_classes)
        self.t_lstm = nn.LSTM(1, hidden_dim, 1, batch_first=True,bias=True,bidirectional=False)
        # self.f_lstm = nn.LSTM(1, hidden_dim, 1, batch_first=True,bias=True,bidirectional=False)
        self.fc = nn.Linear(hidden_dim, 1)
        self.h = int(math.sqrt(in_dim))


    def restruction(self,x,mask=None):
        # if mask is not None:
        #     x = x * (1-mask)
            
        BS,T,F = x.size()
        x1 = x.unsqueeze(1).reshape([BS,1,T,self.h,self.h])
        x1 = self.inc(x1)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        # print(x5.size())
        x_hat = self.up1(x5, x4)
        x_hat = self.up2(x_hat, x3)
        x_hat = self.up3(x_hat, x2)
        x_hat = self.up4(x_hat, x1)
        x_hat = self.outc(x_hat)
        del x1,x2,x3,x4,x5
        # print(x_hat.size())
        x_hat = x_hat.reshape([BS,T,F])
        # x_hat = x_hat * mask + src_x * (1-mask)
        x = x_hat * mask + x * (1-mask)
        return x
         
    def forward(self, x, mask=None):
        # src_x = x
        # if mask is not None:
        #     x = x * (1-mask)
        BS,T,F = x.size()
        x1 = x.unsqueeze(1).reshape([BS,1,T,self.h,self.h])
        x1 = self.inc(x1)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        # print(x5.size())
        x_hat = self.up1(x5, x4)
        x_hat = self.up2(x_hat, x3)
        x_hat = self.up3(x_hat, x2)
        x_hat = self.up4(x_hat, x1)
        x_hat = self.outc(x_hat)
        del x1,x2,x3,x4,x5
        # print(x_hat.size())
        x_hat = x_hat.reshape([BS,T,F])
        # x_hat = x_hat * mask + src_x * (1-mask)
        x = x_hat * mask + x * (1-mask)
        x = x.unsqueeze(-1) # BS,T,F,1
        # x,_ = self.f_lstm(x.reshape(-1,F,1))
        # x = x.reshape([BS,T,F,-1])
        x, _ = self.t_lstm(x.permute(0,2,1,3).reshape(-1,T,1))
        x = x.reshape([BS,F,T,-1]).permute(0,2,1,3)
        # x = torch.cat([t,f],dim=-1) # BS,T,F,D
        x = self.fc(x[:, -1, :]).squeeze()
        return x,x_hat
  
# x = torch.rand([32,1,26,12,12])
# model = UNet3D(1,1)
# x = model(x)
# print(x.size())
