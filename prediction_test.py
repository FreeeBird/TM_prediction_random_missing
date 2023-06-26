
import os

from utils.weighted_loss import AutomaticWeightedLoss, WeightedLoss
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import sys
from utils.early_stop import EarlyStopping
from utils.losses import FusionLoss, WeightL1Loss
sys.path.append(os.getcwd())
import copy
import sys
import time
from args import parse_args
from utils.time_helper import format_sec_tohms, get_datetime_str
import numpy as np
import torch.nn
from tqdm import tqdm
from utils.metrics import ERROR_RATIO, MSE, R2, RMSE
from utils.log_helper import get_logger, save_epoch, save_epoch_test_result, save_result, save_to_excel
from utils.data_helper import *
from utils.model_helper import *
from tensorboardX import SummaryWriter


def test(model,dataloader,num_flows,device,seq_len):
    y_true, y_pred = torch.empty([0,num_flows]), torch.empty([0,  num_flows])
    y_true, y_pred = y_true.to(device), y_pred.to(device)
    x_true, x_pred = torch.empty([0,seq_len,num_flows]),torch.empty([0,seq_len,num_flows])
    MASK = torch.empty([0,seq_len,num_flows])
    x_true, x_pred, MASK = x_true.to(device), x_pred.to(device), MASK.to(device)
    with torch.no_grad():
        model.eval()
        for x, y in dataloader:
            x = x.to(device)
            x,mask = x[:,:,:,0],x[:,:,:,1]
            y = y.to(device)
            y_hat,x_hat = model(x,mask)
            # y_hat = pre_model(x_hat * mask + x * (1-mask))
            y_true = torch.cat((y_true,y), 0)
            y_pred = torch.cat((y_pred,y_hat), 0)
            x_true = torch.cat((x_true, x), 0)
            x_pred = torch.cat((x_pred, x_hat), 0)
            MASK = torch.cat((MASK, mask), 0)
    error_ratio = ERROR_RATIO(x_pred,x_true,MASK)
    mse = MSE(y_pred, y_true)
    rmse = RMSE(y_pred, y_true)
    r2 = R2(y_pred, y_true)
    # print(MASK[0,0])
    return mse,rmse,r2,error_ratio

def test_with_mask(model,dataloader,num_flows,device,seq_len):
    y_true, y_pred = torch.empty([0,num_flows]), torch.empty([0,  num_flows])
    y_true, y_pred = y_true.to(device), y_pred.to(device)
    with torch.no_grad():
        model.eval()
        for x, y in dataloader:
            x = x.to(device)
            x,mask = x[:,:,:,0],x[:,:,:,1]
            y = y.to(device)
            y_hat = model(x,mask)
            y_true = torch.cat((y_true,y), 0)
            y_pred = torch.cat((y_pred,y_hat), 0)
    mse = MSE(y_pred, y_true)
    rmse = RMSE(y_pred, y_true)
    r2 = R2(y_pred, y_true)
    return mse,rmse,r2,mse

def get_dict(missing_ratio = 0.6):
    dicts = {
        0:'dict/LSTM_abilene_26_04-14_11:17:28_dict.pkl',
        0.1:['dict/UNet_LSTM2D_abilene_26_06-08_18:01:18_dict.pkl','dict/UNet_LSTM2D_abilene_26_06-08_19:08:52_dict.pkl','dict/UNet_LSTM2D_abilene_26_06-08_20:22:24_dict.pkl'],
        0.2:['dict/UNet_LSTM2D_abilene_26_06-08_22:43:07_dict.pkl','dict/UNet_LSTM2D_abilene_26_06-09_00:06:44_dict.pkl','dict/UNet_LSTM2D_abilene_26_06-09_01:15:59_dict.pkl'],
        0.3:['dict/UNet_LSTM2D_abilene_26_06-09_11:38:10_dict.pkl','dict/UNet_LSTM2D_abilene_26_06-09_12:49:04_dict.pkl','dict/UNet_LSTM2D_abilene_26_06-09_14:39:39_dict.pkl'],
        0.4:['dict/UNet_LSTM2D_abilene_26_06-09_16:03:14_dict.pkl','dict/UNet_LSTM2D_abilene_26_06-09_17:47:23_dict.pkl','dict/UNet_LSTM2D_abilene_26_06-09_19:01:15_dict.pkl'],
        0.5:['dict/UNet_LSTM2D_abilene_26_06-09_20:58:42_dict.pkl','dict/UNet_LSTM2D_abilene_26_06-09_22:35:50_dict.pkl','dict/UNet_LSTM2D_abilene_26_06-10_02:07:34_dict.pkl'],
        0.6:['dict/UNet_LSTM2D_abilene_26_06-08_13:40:52_dict.pkl','dict/UNet_LSTM2D_abilene_26_06-08_14:25:58_dict.pkl','dict/UNet_LSTM2D_abilene_26_06-08_15:30:58_dict.pkl'],
        0.7:['dict/UNet_LSTM2D_abilene_26_06-10_13:18:32_dict.pkl','dict/UNet_LSTM2D_abilene_26_06-10_14:26:36_dict.pkl','dict/UNet_LSTM2D_abilene_26_06-10_15:44:28_dict.pkl'],
        0.8:['dict/UNet_LSTM2D_abilene_26_06-10_09:56:02_dict.pkl','dict/UNet_LSTM2D_abilene_26_06-10_11:11:27_dict.pkl','dict/UNet_LSTM2D_abilene_26_06-10_12:46:47_dict.pkl'],
        0.9:['dict/UNet_LSTM2D_abilene_26_06-10_01:00:13_dict.pkl','dict/UNet_LSTM2D_abilene_26_06-10_02:35:09_dict.pkl','dict/UNet_LSTM2D_abilene_26_06-10_03:27:18_dict.pkl'],
        0.35:['dict/UNet_LSTM2D_abilene_26_06-11_19:55:18_dict.pkl','dict/UNet_LSTM2D_abilene_26_06-11_20:37:02_dict.pkl','dict/UNet_LSTM2D_abilene_26_06-11_21:39:49_dict.pkl'],
        0.45:['dict/UNet_LSTM2D_abilene_26_06-11_23:16:37_dict.pkl','dict/UNet_LSTM2D_abilene_26_06-12_01:05:31_dict.pkl','dict/UNet_LSTM2D_abilene_26_06-12_03:22:51_dict.pkl'],
        0.55:['dict/UNet_LSTM2D_abilene_26_06-12_10:36:49_dict.pkl','dict/UNet_LSTM2D_abilene_26_06-12_11:55:41_dict.pkl','dict/UNet_LSTM2D_abilene_26_06-12_13:10:38_dict.pkl'],
        0.65:['dict/UNet_LSTM2D_abilene_26_06-12_16:10:08_dict.pkl','dict/UNet_LSTM2D_abilene_26_06-12_17:50:29_dict.pkl','dict/UNet_LSTM2D_abilene_26_06-12_20:04:20_dict.pkl'],
        0.75:['dict/UNet_LSTM2D_abilene_26_06-12_22:00:03_dict.pkl','dict/UNet_LSTM2D_abilene_26_06-12_23:28:12_dict.pkl','dict/UNet_LSTM2D_abilene_26_06-13_00:26:16_dict.pkl'],
        0.16:['dict/UNet_LSTM2D_abilene_26_06-13_12:12:29_dict.pkl','dict/UNet_LSTM2D_abilene_26_06-13_13:10:47_dict.pkl','dict/UNet_LSTM2D_abilene_26_06-13_13:50:15_dict.pkl'],
        0.32:['dict/UNet_LSTM2D_abilene_26_06-13_15:33:45_dict.pkl','dict/UNet_LSTM2D_abilene_26_06-13_16:51:55_dict.pkl','dict/UNet_LSTM2D_abilene_26_06-13_17:34:04_dict.pkl'],
        0.48:['dict/UNet_LSTM2D_abilene_26_06-13_18:40:44_dict.pkl','dict/UNet_LSTM2D_abilene_26_06-13_19:40:48_dict.pkl','dict/UNet_LSTM2D_abilene_26_06-13_20:40:04_dict.pkl'],
        0.64:['dict/UNet_LSTM2D_abilene_26_06-13_22:09:31_dict.pkl','dict/UNet_LSTM2D_abilene_26_06-13_22:49:06_dict.pkl','dict/UNet_LSTM2D_abilene_26_06-13_23:48:23_dict.pkl'],
        0.215:['dict/UNet_LSTM2D_abilene_26_06-14_19:10:30_dict.pkl','dict/UNet_LSTM2D_abilene_26_06-14_21:10:08_dict.pkl','dict/UNet_LSTM2D_abilene_26_06-14_22:14:50_dict.pkl'],
        0.315:['dict/UNet_LSTM2D_abilene_26_06-14_11:27:15_dict.pkl','dict/UNet_LSTM2D_abilene_26_06-14_12:15:57_dict.pkl','dict/UNet_LSTM2D_abilene_26_06-14_13:37:48_dict.pkl'],
    }
    return dicts[missing_ratio]

if __name__ == '__main__':
    # result_xlsx = '/home/liyiyong/TM_Prediction_With_Missing_Data/result.xlsx'
    # sheet_name_xlsx = 'Sheet1'
    # set gpu
    # if args.gpu == 1:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dataset = 'abilene'
    rounds = 3
    seq_len = 26
    # get dataset and num of nodes/flows
    data_path = get_data_path(dataset)
    num_nodes,num_flows = get_dataset_nodes(dataset)
    # all rounds metrices
    ALL_RMSE,ALL_R2 = [],[]
    missing_ratio = 0.1
    target_ratio = 0.1
    std=0.05
    dicts = get_dict(missing_ratio)
    for d in dicts:
        DICT_RMSE = []
        DICT_R2 = []
        model = UNet_LSTM2D(1,bilinear=True,in_dim=144,hidden_dim=64,n_layer=3,dropout=0.2).to(device)
        model.load_state_dict(torch.load(d))
        model.eval()
        for i in range(1,rounds+1):
            dataloader = get_dataloaders(data_path=data_path,train_rate=0.6,seq_len=seq_len,pre_len=1
            ,missing_ratio=target_ratio,missing_index=i,batch_size=32,num_workers=8,imputer=None,random=[False,False,False],test=True,std=std)
            mse,rmse,r2,er = test(model,dataloader['test'],num_flows,device,seq_len)
            save_result(None,d,mse,er,rmse,r2)
            DICT_RMSE.append(rmse.item())
            DICT_R2.append(r2.item())
        print("MEAN_RMSE",np.mean(DICT_RMSE),np.var(DICT_RMSE))
        print("MEAN_R2",np.mean(DICT_R2),np.var(DICT_R2))
        ALL_RMSE.append(np.mean(DICT_RMSE))
        ALL_R2.append(np.mean(DICT_R2))
    print("ALL_MEAN_RMSE",np.mean(ALL_RMSE),np.var(ALL_RMSE))
    print("ALL_MEAN_R2",np.mean(ALL_R2),np.var(ALL_R2))
