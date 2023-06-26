import copy
import sys
import time
import os
sys.path.append(os.getcwd())

from tools.tool import get_logger, get_model, save_result
from train_helper import miss_train_test_model
import numpy as np
import torch.nn
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import argparse
# from tools.early_stop import EarlyStopping
from dataset.helper import get_data_path, get_data_nodes
from model.transformer_fill import Transformer_Fill
from tools.data_process import random_split_dataset, get_dataloader, split_dataset, split_dataset_imputer
from tools.losses import MaskedMSELoss
from tools.metrics import ERROR_RATIO, R2, RMSE


parser = argparse.ArgumentParser()
# parser.add_argument('--model', default='gru', help='train model name')
parser.add_argument('--epochs', default=200, help='epochs')
parser.add_argument('--model', default='KNN_LSTM', help='model name')
parser.add_argument('--dataset', default='abilene', help='chose dataset', choices=['geant', 'abilene'])
parser.add_argument('--gpu', default=0, help='use -1/0/1 chose cpu/gpu:0/gpu:1', choices=[-1, 0, 1])
parser.add_argument('--batch_size', '--bs', default=128, help='batch_size')
parser.add_argument('--learning_rate', '--lr', default=0.0001, help='learning_rate')
parser.add_argument('--seq_len', default=12, help='input history length')
parser.add_argument('--pre_len', default=1, help='prediction length')
parser.add_argument('--dim_model', default=64, help='dimension of embedding vector')
# parser.add_argument('--dim_ff', default=2048, help='dimension of ff')
# parser.add_argument('--num_heads', default=8, help='attention heads')
parser.add_argument('--train_rate', default=0.6, help='')
parser.add_argument('--test_rate', default=0.2, help='')
parser.add_argument('--rnn_layers', default=3, help='rnn layers')
# parser.add_argument('--encoder_layers', default=3, help='encoder layers')
parser.add_argument('--dropout', default=0.2, help='dropout rate')
parser.add_argument('--missing_ratio','--ms', default=0.6, help='missing rate')
parser.add_argument('--early_stop', default=15, help='early stop patient epochs')
parser.add_argument('--rounds', default=3, help='rounds')

args = parser.parse_args()
ts = time.strftime("%m%d%H%M%S", time.localtime())
log_file = "logs/miss_train_log_{}_{}.txt".format(args.model,str(ts))
logger = get_logger(log_file)
if args.gpu == 1:
    os.environ["CUDA_VISIBLE_DEVICES"] = "1,0"
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# param
dataset = args.dataset
data_path = get_data_path(dataset)
num_nodes = get_data_nodes(dataset)
num_flows = num_nodes * num_nodes
# early_stop = EarlyStopping(patience=args.early_stop)


ALL_RMSE = []
ALL_ER = []
ALL_R2 = []
for i in range(1,args.rounds+1):
    from tensorboardX import SummaryWriter
    # 定义Summary_Writer
    writer = SummaryWriter('./miss_train_result',flush_secs=20)  
    # hyper param
    epoch = args.epochs
    batch_size = args.batch_size
    lr = args.learning_rate
    seq_len = args.seq_len
    pre_len = args.pre_len
    dim_model = args.dim_model
    train_rate = args.train_rate
    test_rate = args.test_rate
    dropout = args.dropout
    args.num_flows = num_flows
    missing_ratio = args.missing_ratio
    ################# data
    # split dataset
    train_x, train_y, val_x, val_y, test_x, test_y, max_value = split_dataset_imputer(data_path, train_rate=train_rate,
                                                                            seq_len=seq_len,
                                                                            predict_len=pre_len, 
                                                                            missing_ratio=missing_ratio,missing_index=i,
                                                                            imputation='knn'
                                                                            )
    # dataloader
    train_loader = get_dataloader(train_x, train_y)
    val_loader = get_dataloader(val_x, val_y)
    test_loader = get_dataloader(test_x, test_y, shuffle=False, pip_memory=False)

    ################# model
    # model = LSTM_FILL(num_flows, dim_model, 3)
    model = get_model(args.model, args)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.MSELoss()

    print(args)

    # test_y = test_y*max_data
    time_start = time.time()
    train_losses = []
    MIN_MSE = 1e5
    best_model_dict = model.state_dict()
    logger.info(time.strftime("%m-%d_%H:%M:%S", time.localtime()))
    logger.info('Start training on ' + args.model)
    ###### train ####
    for e in range(1, epoch + 1):
        model.train()
        train_loss = 0.0
        for x, y in tqdm(train_loader,ncols=80,position=0):
            x = x.to(device)
            y = y.to(device)
            x, mask = torch.split(x, [1, 1], dim=-1)
            y_hat = model(x,mask)
            loss = criterion(y_hat, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss = train_loss / len(train_loader)
        train_losses.append(train_loss)
        # model val
        eval_loss = 0.0
        with torch.no_grad():
            model.eval()
            for x, y in val_loader:
                x = x.to(device)
                y = y.to(device)
                x, mask = torch.split(x, [1, 1], dim=-1)
                y_hat = model(x,mask)
                loss = criterion(y_hat, y)
                eval_loss += loss.item()
        eval_loss = eval_loss / len(val_loader)
        print('Epoch:{}'.format(e),
            'train_mse:{:.6}'.format(train_loss),
            'val_mse:{:.6}'.format(eval_loss),
            )
        if eval_loss < MIN_MSE:
            MIN_MSE = eval_loss
            best_model_dict = copy.deepcopy(model.state_dict())
            # print('*NEW MIN VAL LOSS')
            logger.info("*NEW MIN VAL LOSS*")
            tmse,trmse,tr2 = miss_train_test_model(model,test_loader,num_flows,criterion,device)
            save_result(logger,'epoch {}'.format(e),tmse,0.,trmse,tr2)
    time_end = time.time()
    ts = time.strftime("%m-%d_%H:%M:%S", time.localtime())
    print(ts)
    dict_name = 'dict/' + model.__class__.__name__ + "_" + args.dataset + "_" + str(seq_len) + "-" + str(
                pre_len) + "_" + ts + '_dict.pkl'
    torch.save(best_model_dict,dict_name)
    print((time_end - time_start) / 3600, 'h')
    print(args)
    logger.info(ts)
    logger.info(dict_name)
    logger.info(str((time_end - time_start) / 3600) + ' h')
    logger.info(str(args))
    ########## test ###########
    model.load_state_dict(best_model_dict)
    test_start = time.time()
    y_true, y_pred = torch.empty([0,num_flows]), torch.empty([0,  num_flows])
    y_true, y_pred = y_true.to(device), y_pred.to(device)
    with torch.no_grad():
        model.eval()
        for x, y in test_loader:
            x = x.to(device)
            y = y.to(device)
            x, mask = torch.split(x, [1, 1], dim=-1)
            y_hat = model(x,mask)
            y_true = torch.cat((y_true,y), 0)
            y_pred = torch.cat((y_pred,y_hat), 0)
    mse = criterion(y_pred, y_true)
    rmse = RMSE(y_pred, y_true)
    r2 = R2(y_pred, y_true)
    print('TEST RESULT:',
        'mse:{:.6}'.format(mse),
        'rmse:{:.6}'.format(rmse),
        'r2:{:.6}'.format(r2),
        )
    test_end = time.time()
    print("test time: ", (time_end - time_start), ' S')
    logger.info("test time: " +str( (time_end - time_start)) + ' S')
    save_result(logger,dict_name,mse,0.,rmse,r2)
    ALL_RMSE.append(rmse.cpu())
    ALL_R2.append(r2.cpu())
    print("ALL_RMSE",np.mean(ALL_RMSE),np.var(ALL_RMSE))
    print("ALL_R2",np.mean(ALL_R2),np.var(ALL_R2))
print("MEAN_RMSE",np.mean(ALL_RMSE),np.var(ALL_RMSE))
print("MEAN_R2",np.mean(ALL_R2),np.var(ALL_R2))
logger.info("MEAN_RMSE:")
logger.info(np.mean(ALL_RMSE))
logger.info(np.var(ALL_RMSE))
logger.info("ALL_R2:")
logger.info(np.mean(ALL_R2))
logger.info(np.var(ALL_R2))
logger.info(log_file)
