'''
Author: FreeeBird
Date: 2022-04-21 19:57:46
LastEditTime: 2022-04-22 10:43:00
LastEditors: FreeeBird
Description: 
FilePath: /TM_Prediction_With_Missing_Data/utils/losses.py
'''
import torch
import torch.nn as nn

from utils.ssim import MS_SSIM


class WeightL1Loss(torch.nn.Module):
    def __init__(self):
        super(WeightL1Loss,self).__init__()
        self.loss = torch.nn.L1Loss(reduction='none')
        
    def forward(self,x,y,mask):
        loss = self.loss(x,y)
        return torch.mean(0.9 * loss * mask + 0.1*(1-mask)*loss)

class FusionLoss(torch.nn.Module):
    def __init__(self):
        super(FusionLoss,self).__init__()
        self.wl1loss = WeightL1Loss()
        self.msssim = MS_SSIM(win_size=1, win_sigma=1.5, data_range=1, size_average=True, channel=1)
        
    def forward(self,x,y,mask):
        l1loss = self.wl1loss(x,y,mask)
        msssimloss = 1- self.msssim(x.unsqueeze(1),y.unsqueeze(1))
        return 0.8 * msssimloss + 0.2 * l1loss

class MaskedMSELoss(torch.nn.Module):
    def __init__(self):
        super(MaskedMSELoss, self).__init__()

    def forward(self, pred, target, mask):
        loss = torch.sum(((pred-target)*mask)**2.0) / torch.sum(mask)
        return loss


class FusionMSELoss(torch.nn.Module):
    def __init__(self):
        super(FusionMSELoss, self).__init__()
        self.mse = torch.nn.MSELoss()
        # self.e = torch.nn.Parameter(torch.tensor([0.5]),requires_grad=True).cuda()
        self.e = 0.5

    def forward(self, target,pred, y_hat,y, mask=None):

        y_loss = self.mse(y_hat,y)
        if mask is None:
            return y_loss
        x_loss = torch.sum(((pred-target)*mask)**2.0) / torch.sum(mask)
        return x_loss*self.e + y_loss*(1-self.e)
