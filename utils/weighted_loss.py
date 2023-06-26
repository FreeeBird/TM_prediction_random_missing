'''
Author: FreeeBird
Date: 2022-04-27 10:48:00
LastEditTime: 2022-04-27 18:09:11
LastEditors: FreeeBird
Description: 
FilePath: /TM_Prediction_With_Missing_Data/utils/weighted_loss.py
'''
import torch
import torch.nn as nn

class AutomaticWeightedLoss(nn.Module):
    """automatically weighted multi-task loss
    Params：
        num: int,the number of loss
        x: multi-task loss
    Examples：
        loss1=1
        loss2=2
        awl = AutomaticWeightedLoss(2)
        loss_sum = awl(loss1, loss2)
    """
    def __init__(self, num=2):
        super(AutomaticWeightedLoss, self).__init__()
        params = torch.ones(num, requires_grad=True)
        self.params = torch.nn.Parameter(params)

    def forward(self, *x):
        loss_sum = 0
        for i, loss in enumerate(x):
            loss_sum += 0.5 / (self.params[i] ** 2) * loss + torch.log(self.params[i] ** 2)
        return loss_sum



class WeightedLoss(nn.Module):
    """automatically weighted multi-task loss
    Params：
        num: int,the number of loss
        x: multi-task loss
    Examples：
        loss1=1
        loss2=2
        awl = AutomaticWeightedLoss(2)
        loss_sum = awl(loss1, loss2)
    """
    def __init__(self, num=1):
        super(WeightedLoss, self).__init__()
        params = torch.ones(num, requires_grad=True)
        params = params*0.5
        self.params = torch.nn.Parameter(params)

    def forward(self, x1,x2):
        loss_sum  = self.params[0] * x1 + (1-self.params[0])* x2
        return loss_sum



if __name__ == '__main__':
    awl = AutomaticWeightedLoss(2)
    print(awl.parameters())