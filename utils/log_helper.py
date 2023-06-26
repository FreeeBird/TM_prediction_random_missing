'''
Author: FreeeBird
Date: 2022-04-13 15:59:13
LastEditTime: 2022-04-28 11:02:22
LastEditors: FreeeBird
Description: 
FilePath: /TM_Prediction_With_Missing_Data/utils/log_helper.py
'''
from asyncio.log import logger
import os
import sys

from utils.excel_helper import write_excel_xlsx
sys.path.append(os.getcwd())
import logging
import numpy as np

# train_log_filename = "train_log.txt"
# train_log_filepath = os.path.join(os.path.curdir, train_log_filename)
 
# result_xlsx = '/home/liyiyong/TM_Prediction_With_Missing_Data/result.xlsx'
# sheet_name_xlsx = 'Sheet1'

def save_epoch(logger,epoch=1,epochs=200,train_mse=0.,val_mse=0.,val_er=0.):
    train_log_txt = "Epoch:[{}/{}]\t train_loss={:.6}\t val_loss={:.6}\t val_er={:.6}".format(epoch,epochs,train_mse,val_mse,val_er)
    logger.info(train_log_txt)

def save_epoch_test_result(logger,epoch=1,epochs=200,mse=0.,er=0.,rmse=0.,r2=0.):
    train_log_txt_formatter = "[Test Epoch {epoch}/{epochs}] [MSE] {MSE:.6f} [RMSE] {RMSE:.6f} [R2] {R2:.6f}  [ER] {ER:.6f}\n"
    to_write = train_log_txt_formatter.format(epoch=epoch,epochs=epochs,MSE=mse,RMSE=rmse,R2=r2,ER=er)
    logger.info(to_write)

def save_result(logger,dict_name='',mse=0.,er=0.,rmse=0.,r2=0.):
    train_log_txt_formatter = "[Dict_name] {Dict_name} [MSE] {MSE:.6f} [RMSE] {RMSE:.6f} [R2] {R2:.6f}  [ER] {ER:.6f}\n"
    to_write = train_log_txt_formatter.format(Dict_name = dict_name,MSE=mse,RMSE=rmse,R2=r2,ER=er)
    if logger is not None:
        logger.info(to_write)
    

def save_to_excel(path,arg,log_file,RMSE,R2,ER):
    # Model	Seq Len	Pre Len	Missing Ratio	RMSE	R²	Error Ratio	Log	Args	RMSE1	R²1	ER1	RMSE2	R²2	ER2	RMSE3	R²3
    value = [arg.model,arg.seq_len,arg.pre_len,arg.missing_ratio,np.mean(RMSE),np.mean(R2),np.mean(ER),
              log_file,str(arg),RMSE[0],R2[0],ER[0],RMSE[1],R2[1],ER[1],RMSE[2],R2[2],ER[2]]
    write_excel_xlsx(path,[value])
    logger.info('Test results had save to ' + path)
 

 
def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])
 
    fh = logging.FileHandler(filename, "x")
    fh.setFormatter(formatter)
    logger.addHandler(fh)
 
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)
 
    return logger



