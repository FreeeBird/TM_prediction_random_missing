'''
Author: FreeeBird
Date: 2022-04-18 16:01:41
LastEditTime: 2022-04-22 15:00:06
LastEditors: FreeeBird
Description: 
FilePath: /TM_Prediction_With_Missing_Data/restruction_train.py
'''
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
import sys
from utils.early_stop import EarlyStopping
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
from utils.log_helper import get_logger, save_epoch, save_result
from utils.data_helper import *
from utils.model_helper import *
criterion = torch.nn.MSELoss()
# res_loss_func = WeightL1Loss()


def train(model,optimizer,criterion,dataloader,e):
    train_loss = 0.0
    model.train()
    for x, y in tqdm(dataloader,ncols=80,position=0):
        x = x.to(device)
        
        if(x.size()[-1]<3):
            x,mask = x[:,:,:,0],x[:,:,:,1]
            x_imputed = x*(1-mask)
        else:
            x,mask,x_imputed = x[:,:,:,0],x[:,:,:,1],x[:,:,:,2]
        x_res = model(x,mask).squeeze(1)
        # x_hat = x_imputed
        x_hat = x_imputed
        x_hat[:,1:-1] = x_res
        loss = criterion(x_hat*mask, x*mask)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    train_loss = train_loss / len(dataloader)
    return train_loss

def validate(model,criterion,dataloader,e):
    val_loss = 0.0
    # ers = 0.0
    with torch.no_grad():
        model.eval()
        for x, y in dataloader:
            x = x.to(device)
            if(x.size()[-1]<3):
                x,mask = x[:,:,:,0],x[:,:,:,1]
                x_imputed = x*(1-mask)
            else:
                x,mask,x_imputed = x[:,:,:,0],x[:,:,:,1],x[:,:,:,2]
            
            x_res = model(x,mask).squeeze(1)
            x_hat = x_imputed
            x_hat[:,1:-1] = x_res
            loss = criterion(x_hat*mask, x*mask)
            val_loss += loss.item()
            # ers += er
    val_loss = val_loss / len(dataloader)
    return val_loss

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
            if(x.size()[-1]<3):
                x,mask = x[:,:,:,0],x[:,:,:,1]
                x_imputed = x*(1-mask)
            else:
                x,mask,x_imputed = x[:,:,:,0],x[:,:,:,1],x[:,:,:,2]
            x_res = model(x,mask).squeeze(1)
            x_hat = x_imputed
            x_hat[:,1:-1] = x_res
            x_true = torch.cat((x_true, x), 0)
            x_pred = torch.cat((x_pred, x_hat), 0)
            MASK = torch.cat((MASK, mask), 0)
    error_ratio = ERROR_RATIO(x_pred,x_true,MASK)
    mse = MSE(x_pred, x_true)
    rmse = RMSE(x_pred, x_true)
    r2 = R2(x_pred, x_true)
    return mse,rmse,r2,error_ratio


if __name__ == '__main__':
    result_xlsx = '/home/liyiyong/TM_Prediction_With_Missing_Data/restruction_result.xlsx'
    sheet_name_xlsx = 'Sheet1'
    args = parse_args()
    ts = get_datetime_str()
    log_file = 'logs/'+__file__.split('/')[-1]+"_log_{}_{}.txt".format(args.model,str(ts))
    logger = get_logger(log_file)
    # set gpu
    if args.gpu == 1:
        os.environ["CUDA_VISIBLE_DEVICES"] = "1,0"
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # get dataset and num of nodes/flows
    data_path = get_data_path(args.dataset)
    num_nodes,num_flows = get_dataset_nodes(args.dataset)
    # all rounds metrices
    ALL_RMSE,ALL_ER,ALL_R2 = [],[],[]
    dict_names = []
    for i in range(1,args.rounds+1):
        # tdw_path = args.tensorboard_dir + log_file + '_' + str(i)
        early_stop = EarlyStopping(patience=args.early_stop, logger=logger)
        # 定义Summary_Writer
        # writer = SummaryWriter(tdw_path,flush_secs=20)  
        # dataloader
        dataloader = get_dataloaders(data_path=data_path,train_rate=args.train_rate,seq_len=args.seq_len,pre_len=args.pre_len
        ,missing_ratio=args.missing_ratio,missing_index=i,batch_size=args.batch_size,num_workers=args.cpu,
        imputer=None,std=args.std)

        # model
        model = get_model(args)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
        

        print(args)

        time_start = time.time()
        train_losses = []
        val_losses = []
        logger.info('Training on ' + args.model)
        ###### train ####
        for e in range(1, args.epochs + 1):
            train_loss = train(model,optimizer,criterion,dataloader['train'],e)
            val_loss = validate(model,criterion,dataloader['val'],e)
            train_losses.append(train_loss)
            save_epoch(logger,e,args.epochs,train_loss,val_loss)
            # writer.add_scalar('train_loss',train_loss,e)
            # writer.add_scalar('val_loss',val_loss,e)
            es,new_high = early_stop(val_loss)
            if new_high:
                early_stop.save_model_dict(copy.deepcopy(model.state_dict()))
                logger.info("*NEW MIN VAL LOSS*")
                # tmse,trmse,tr2,er2 = test(model,dataloader['test'],num_flows,device,args.seq_len)
                # save_epoch_test_result(logger,e,args.epochs,tmse,er2,trmse,tr2)
            if es:
                break
        time_end = time.time()
        cost_time = format_sec_tohms(time_end - time_start)
        ts = get_datetime_str()
        dict_name = 'dict/' + model.__class__.__name__ + "_" + args.dataset + "_" + str(args.seq_len) + "_" + ts + '_dict.pkl'
        torch.save(early_stop.get_best_model_dict(),dict_name)
        dict_names.append(dict_name)
        logger.info(ts)
        logger.info(dict_name)
        logger.info(cost_time)
        logger.info(str(args))
        ########## test ###########
        model.load_state_dict(early_stop.get_best_model_dict())
        mse,rmse,r2,er = test(model,dataloader['test'],num_flows,device,args.seq_len)
        save_result(logger,dict_name,mse,er,rmse,r2)
        ALL_RMSE.append(rmse.item())
        ALL_R2.append(r2.item())
        ALL_ER.append(er.item())
        print("ALL_RMSE",np.mean(ALL_RMSE),np.var(ALL_RMSE))
        print("ALL_R2",np.mean(ALL_R2),np.var(ALL_R2))
    print("MEAN_RMSE",np.mean(ALL_RMSE),np.var(ALL_RMSE))
    print("MEAN_R2",np.mean(ALL_R2),np.var(ALL_R2))
    # save_to_excel(result_xlsx,args,log_file,ALL_RMSE,ALL_R2,ALL_ER)
    logger.info("MEAN_RMSE:")
    logger.info(np.mean(ALL_RMSE))
    logger.info("MEAN_R2:")
    logger.info(np.mean(ALL_R2))
    logger.info(log_file)
    print(dict_names)
