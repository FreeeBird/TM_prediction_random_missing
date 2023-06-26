'''
Author: FreeeBird
Date: 2022-04-13 16:11:06
LastEditTime: 2022-04-23 20:07:21
LastEditors: FreeeBird
Description: 
FilePath: /TM_Prediction_With_Missing_Data/utils/early_stop.py
'''
import numpy as np
import torch


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, patience=7, verbose=False, delta=0, logger=None):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.best_dict = {}
        self.logger = logger
        self.new_high = False

    def __call__(self, val_loss):
        # if self.patience<=0:
        #     return False,False
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.new_high = True
            # self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.new_high = False
            self.logger.info(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.patience>0 and self.counter >= self.patience :
                self.early_stop = True
        else:
            self.best_score = score
            self.new_high = True
            self.counter = 0
            # self.save_checkpoint(val_loss, model)
            # self.logger.info(f'EarlyStopping counter: {self.counter}')
        return self.early_stop,self.new_high
    
    def zero_count(self):
        self.counter = 0
        self.early_stop = False
    
    def get_best_model_dict(self):
        return self.best_dict
    
    def save_model_dict(self,model_dict):
        self.best_dict = model_dict

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), 'checkpoint.pt') 
        self.val_loss_min = val_loss
