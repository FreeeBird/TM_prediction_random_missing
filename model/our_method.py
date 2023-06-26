

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.LSTNet import LSTNet
from model.MTGNN import gtnet

class DoubleConv3(nn.Module):
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


class Down3(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels,mp_straide=2):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool3d(mp_straide),
            DoubleConv3(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up3(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=False):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
            self.conv = DoubleConv3(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose3d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv3(in_channels, out_channels)

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
    

class OutConv3(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv3, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


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


class UNet3D(nn.Module):
    def __init__(self,in_dim, n_channels=1, n_classes=1, bilinear=False):
        super(UNet3D, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv3(n_channels, 32)
        self.down1 = Down3(32, 64)
        self.down2 = Down3(64, 128)
        self.down3 = Down3(128, 256)
        factor = 2 if bilinear else 1
        self.down4 = Down3(256, 512 // factor,1)
        self.up1 = Up3(512, 256 // factor, bilinear)
        self.up2 = Up3(256, 128 // factor, bilinear)
        self.up3 = Up3(128, 64 // factor, bilinear)
        self.up4 = Up3(64, 32, bilinear)
        self.outc = OutConv3(32, n_classes)
        self.fc = nn.Linear(in_dim, in_dim)
        self.h = int(math.sqrt(in_dim))
        self.sigmoid = nn.Sigmoid()
        
    # def restruction(self,x_res,fx,bx):
    #     bx = bx.flip(1)
    #     bx = bx + x_res
    #     bx[1:] = bx[1:] + fx[:-1]
    #     return bx

    def forward(self, x):
        
        BS,_,T,F = x.size()
        x1 = x.reshape([BS,1,T,self.h,self.h])
        # print(x1.size())
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
        return x_hat

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
    

  
""" 
mc_module -> Matrix Completion Module
mp_module -> Matrix Prediction Module
"""
class Proposed_Method(nn.Module):
    def __init__(self,mc_module="unet",mp_module="lstm2d", n_channels=1, bilinear=True,in_dim=144, hidden_dim=144, 
                 n_layer=3,dropout=0.2,timestep=26,args=None):
        super(Proposed_Method, self).__init__()
        self.in_chan = n_channels
        self.mc = self.get_mc_module(mc_module,timestep,hidden_dim,args)
        self.mp = self.get_mp_module(mp_module,timestep,hidden_dim,dropout,args)
        # self.unet = UNet(n_channels,1,bilinear)
        # self.t_lstm = nn.LSTM(1, hidden_dim, 1, batch_first=True,bias=True,bidirectional=False)
        # self.f_lstm = nn.LSTM(1, hidden_dim, 1, batch_first=True,bias=True,bidirectional=False)
        # self.fc = nn.Linear(2*hidden_dim, 1)
        # self.node = int(math.sqrt(in_dim))
        
        
    def get_mc_module(self,mou_name = "unet",timestep=26,hidden_dim=144,dropout=0.2,args=None):
        model = None
        if mou_name=="autoencoder":
            model = AutoEncoder(timestep=timestep, hidden_dim=hidden_dim)
        if mou_name == 'unet':
            model = UNet(1,1,True)
        if mou_name == 'unet3d':
            model = UNet3D(hidden_dim)
        return model
    
    def get_mp_module(self,mou_name="lstm2d",timestep=26,hidden_dim=144,dropout=0.2,args=None):
        model = None
        if mou_name=="lstm2d":
            model = LSTM2D(hidden_dim=hidden_dim)
        if mou_name == 'lstnet':
            model = LSTNet(flows=144,seq_len=timestep,pre_len=1,hidCNN=64,hidRNN=64,hidSkip=10,CNN_kernel=6,skip=2)
        if mou_name == 'mtgnn':
            model = gtnet(gcn_true=True, buildA_true=True, gcn_depth=2, num_nodes=12, device='cuda', predefined_A=args.m_adj, static_feat=None, 
    dropout=0.2, subgraph_size=2, node_dim=64, dilation_exponential=1, conv_channels=64, residual_channels=64, 
    skip_channels=64, end_channels=64, seq_length=26, in_dim=12, out_dim=12, layers=3, propalpha=0.05, tanhalpha=3, layer_norm_affline=True)
        return model
    
    
    def forward(self, x, mask=None):
        if mask is not None:
            x = x * (1-mask) 
        BS,T,F = x.size()
        if self.in_chan == 1:
            x_hat = x.unsqueeze(1) # BS,1,T,F
        else:
            x_hat = torch.stack([x,mask],dim=1)
        
        x_hat = self.mc(x_hat)
        x_hat = x_hat.reshape([BS,T,F])
        x = x_hat * mask + x * (1-mask)
        x = x.unsqueeze(-1) # BS,T,F,1
        x = self.mp(x) # BS,F
        # f,_ = self.f_lstm(x.reshape(-1,F,1))
        # f = f.reshape([BS,T,F,-1])
        # t, _ = self.t_lstm(x.permute(0,2,1,3).reshape(-1,T,1))
        # t = t.reshape([BS,F,T,-1]).permute(0,2,1,3)
        # x = torch.cat([t,f],dim=-1) # BS,T,F,D
        # x = self.fc(x[:, -1, :]).squeeze()
        return x,x_hat
    

class AutoEncoder(nn.Module):
    def __init__(self, timestep=26, hidden_dim=144):
        super().__init__()
        # [b, 784]
        self.encoder = nn.Sequential(
            nn.Linear(timestep*hidden_dim,1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024,512),
            nn.ReLU(inplace=True),
            nn.Linear(512,64),
            nn.ReLU(inplace=True)
        )   
        self.decoder = nn.Sequential(
            nn.Linear(64,512),
            nn.ReLU(inplace=True),
            nn.Linear(512,1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024,timestep*hidden_dim),
            nn.Sigmoid(),
        )

    def forward(self, x):
        #BS,T,F
        _,_,T,F = x.size()
        x = x.flatten(1)
        x = self.encoder(x)
        x = self.decoder(x)
        x = x.view(-1,1,T,F)
        return x
    
    
class LSTM2D(nn.Module):
    def __init__(self, in_dim=1, hidden_dim=144, n_layer=1,dropout=0.2):
        super(LSTM2D, self).__init__()
        self.t_lstm = nn.LSTM(1, hidden_dim, 1, batch_first=True,bias=True,bidirectional=False)
        self.f_lstm = nn.LSTM(1, hidden_dim, 1, batch_first=True,bias=True,bidirectional=False)
        self.fc = nn.Linear(2*hidden_dim, 1)

    def forward(self, x):
        BS,T,F,_ = x.size()
        # BS,T,F,1
        f,_ = self.f_lstm(x.reshape(-1,F,1))
        f = f.reshape([BS,T,F,-1])
        t, _ = self.t_lstm(x.permute(0,2,1,3).reshape(-1,T,1))
        t = t.reshape([BS,F,T,-1]).permute(0,2,1,3)
        x = torch.cat([t,f],dim=-1) # BS,T,F,D
        x = self.fc(x[:, -1, :]).squeeze()
        return x
    

if __name__ == '__main__':
    model = Proposed_Method(mc_module="autoencoder",mp_module="lstm2d", n_channels=1, bilinear=True,in_dim=144, hidden_dim=144,
                            n_layer=3,dropout=0.2,timestep=26)
    x = torch.zeros([32,26,144])
    mask = torch.ones([32,26,144])
    x,x_hat = model(x,mask)
    print(x.size())
    print(x_hat.size())
    # x = torch.zeros([32,26,529])
    # x = model(x)
    # print(x.size())
    pass
