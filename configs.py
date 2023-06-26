'''
Author: FreeeBird
Date: 2022-04-13 16:32:42
LastEditTime: 2022-11-22 21:59:41
LastEditors: FreeeBird
Description: 
FilePath: /TM_Prediction_With_Missing_Data/configs.py
'''
import os
import torch

class Config(dict):
    def __init__(self, **kwargs):
        """
        Initialize an instance of this class.

        Args:

        """
        super().__init__(**kwargs)
        for key, value in kwargs.items():
            setattr(self, key, value)

    def set(self, key, value):
        """
        Sets the value to the value.

        Args:
            key: (str):
            value:
        """
        self[key] = value
        setattr(self, key, value)


Models = ['LSTM','LSTM2D','gcrint','UNet_LSTM','UNet3D_LSTM2D','UNet_LSTM2D','LSTNet','MTGNN']

config = Config(
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    gpu=0,
    cpu=os.cpu_count(),
    model = 'UNet3D_LSTMT',
    mc_mod = 'unet3d', # for MP
    mp_mod = 'mtgnn', # for MP
    bilinear = True,
    kernel_size = 5,
    in_chan = 1,
    dataset = 'abilene', # abilene or geant
    epochs=200,
    batch_size=32,
    learning_rate=0.0001,
    seq_len=26,  # previous timestamps to use
    pre_len=1,  # number of timestamps to predict
    dim_model = 64,
    rounds = 3,
    heads = 1,
    dim_ff = 512,
    train_rate = 0.6,
    test_rate=0.2,
    rnn_layers =3,
    encoder_layers =1,
    dropout = 0.2,
    missing_ratio = 0.4,
    std = 0.05,
    early_stop = 15,
    flow = 144, # for abilene
    # flow = 529, # for geant
    lw=0.5,
    output_dir = '/home/liyiyong/TM_Prediction_With_Missing_Data/logs',
    log_dir = '/home/liyiyong/TM_Prediction_With_Missing_Data/output',
    tensorboard_dir = '/home/liyiyong/TM_Prediction_With_Missing_Data/tensorboards',
    test_during_training=True
)