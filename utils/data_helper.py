'''
Author: FreeeBird
Date: 2022-04-13 16:12:55
LastEditTime: 2022-11-24 15:47:39
LastEditors: FreeeBird
Description: 
FilePath: /TM_Prediction_With_Missing_Data/utils/data_helper.py
'''
from math import sqrt
import os
import random
import sys
sys.path.append(os.getcwd())
import torch
import numpy as np
from torch.utils.data import DataLoader
from utils.data_process import np_to_tensor_dataset, split_dataset, split_dataset_with_imputer, split_dataset_with_missing
from scipy.stats import truncnorm

def get_truncated_normal(mean=0, sd=1, low=0, upp=10):
    return truncnorm(
        (low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd)

def get_data_path(dataset='abilene'):
    dataset_path = os.path.dirname(os.path.dirname(__file__))
    data_path = None
    if dataset == 'geant':
        data_path = 'geant.npy'
    elif dataset == 'abilene':
        data_path = 'abilene.npy'
    elif dataset == 'abilene2':
        data_path = 'abilene2.npy'
    elif dataset == 'abilene_back':
        data_path = 'abilene_back.npy'
    data_path = os.path.join(dataset_path,'dataset',data_path)
    if data_path is None:
        print("Can not get the {dataset} dataset.".format(dataset))
        print("So get the Abilene dataset.")
        return get_data_path()
    print("Get the {d} dataset at {p}.".format(d=dataset,p=data_path))
    return data_path


def get_dataset_nodes(dataset='geant'):
    nodes = 12
    if dataset == 'geant':
        nodes = 23
    elif dataset == 'abilene':
        nodes = 12
    elif dataset == 'abilene2':
        nodes = 12
    return nodes,nodes**2


def get_adj_matrix(dataset='abilene'):
    adj_path = ''
    if dataset == 'geant':
        adj_path = '/home/liyiyong/flow-wise-prediction/dataset/geant_adj.npy'
    elif dataset == 'abilene':
        adj_path = '/home/liyiyong/flow-wise-prediction/dataset/abilene_adj.npy'
    elif dataset == 'abilene2':
        adj_path = '/home/liyiyong/flow-wise-prediction/dataset/abilene_adj.npy'
    return adj_path


# 1 is missing
def random_missing_2d(row=12, col=144, missing_ratio=0.6):
    missing = np.zeros((row,col))
    for i in range(row):
        tmp = np.zeros(col)
        tmp[:int(col * missing_ratio)] = 1
        np.random.shuffle(tmp)
        missing[i] = tmp
    return missing

def missing_node_2d(len=100,flows=144,nodes=12,miss_node_num=1):
    missing_matrix = np.zeros((len,flows))
    missing_matrix = np.reshape(missing_matrix,[len,nodes,nodes])
    node_list = np.arange(12)
    missing_node = random.sample(list(node_list),miss_node_num)
    for i in range(len):
        tmp = np.zeros([nodes,nodes])
        for mn in missing_node:
            tmp[mn] = 1
            tmp[:,mn] = 1
        missing_matrix[i] = tmp
    return missing_matrix.reshape(len,flows)

def normal_missing_2d(len=26,flows=144,mean_ratio=0.2,std=0.05):
    # mask_rates = np.random.normal(loc = mean_ratio , scale = std,size=len)
    gen = get_truncated_normal(mean_ratio,std,low=mean_ratio-std,upp=mean_ratio+std)
    mask_rates = gen.rvs(len)
    missing_matrix = np.zeros((len,flows))
    for i in range(len):
        tmp = np.zeros(flows)
        tmp[:int(flows * mask_rates[i])] = 1
        np.random.shuffle(tmp)
        missing_matrix[i] = tmp
    return missing_matrix

# random masking with fixed ratio
def gen_missing_matrix(dataset='abilene',missing_ratio=0.6,counts=3):
    data_path = get_data_path(dataset)
    data = np.load(data_path)
    r,c = data.shape
    dataset_path = os.path.dirname(data_path)
    for i in range(1,counts+1):
        missing_matrix = random_missing_2d(r,c,missing_ratio)
        mm_path = dataset + "_missing_"+ str(missing_ratio) + "_" + str(i)
        np.save(os.path.join(dataset_path,mm_path),missing_matrix)

#  gen missing matrix in normal
def gen_normal_missing_matrix(dataset='abilene',mean_ratio=0.2,std=0.05,counts=3):
    data_path = get_data_path(dataset)
    data = np.load(data_path)
    len,flows = data.shape
    dataset_path = os.path.dirname(data_path)
    
    for i in range(1,counts+1):
        if std>0:
            missing_matrix = normal_missing_2d(len,flows,mean_ratio=mean_ratio,std=std)
            mm_path = dataset + "_missing_"+ str(mean_ratio) + "+-" + str(std) + "_" + str(i)
        else:
            missing_matrix = random_missing_2d(len,flows,mean_ratio)
            mm_path = dataset + "_missing_"+ str(mean_ratio) + "_" + str(i)
        # np.save(os.path.join(dataset_path,mm_path),missing_matrix)

# cross masking strategy
def gen_missing_nodes_matrix(dataset='abilene',missing_node_num = 1,counts=3):
    ratio = 0.16 # missing 1 node ~= missing 16% data
    data_path = get_data_path(dataset)
    data = np.load(data_path)
    len,flows = data.shape
    nodes = int(sqrt(flows))
    dataset_path = os.path.dirname(data_path)
    for i in range(1,counts+1):
        missing_matrix = missing_node_2d(len,flows,nodes=nodes,miss_node_num=missing_node_num)
        mm_path = dataset + "_missing_"+ str(ratio*missing_node_num) + "_" + str(i)
        np.save(os.path.join(dataset_path,mm_path),missing_matrix)

# get dataloader
def get_dataloaders(data_path=None,train_rate=0.6,seq_len=26,pre_len=1,missing_ratio=0.6,
                    missing_index=1,batch_size=64,num_workers=8,
                    imputer=None,random=None,test=False,std=0):
    dataloaders = {}
    if imputer !=None:
        train_x, train_y, val_x, val_y, test_x, test_y, max_value,er = split_dataset_with_imputer(data_path, train_rate=train_rate,
                                                                                seq_len=seq_len,
                                                                                predict_len=pre_len, 
                                                                                missing_ratio=missing_ratio,missing_index=missing_index,
                                                                                test=test,imputer=imputer,std=std
                                                                                )
        dataloaders['er'] = er
        dataloaders['time'] = er
    elif missing_ratio > 0:
        train_x, train_y, val_x, val_y, test_x, test_y, max_value = split_dataset_with_missing(data_path, train_rate=train_rate,
                                                                                seq_len=seq_len,
                                                                                predict_len=pre_len, 
                                                                                missing_ratio=missing_ratio,missing_index=missing_index,
                                                                                random_train=random,std=std
                                                                                )
    else:
        train_x, train_y, val_x, val_y, test_x, test_y, max_value = split_dataset(data_path, train_rate=train_rate,
                                                                                seq_len=seq_len,
                                                                                predict_len=pre_len)
    dataset = np_to_tensor_dataset(train_x, train_y)
    dataloaders['train'] = DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=num_workers)
    dataset = np_to_tensor_dataset(val_x, val_y)
    dataloaders['val'] = DataLoader(dataset, batch_size=batch_size, shuffle=False, pin_memory=False, num_workers=num_workers)
    dataset = np_to_tensor_dataset(test_x, test_y)
    dataloaders['test'] = DataLoader(dataset, batch_size=batch_size, shuffle=False, pin_memory=False, num_workers=num_workers)
    return dataloaders
from torch.utils.data import TensorDataset, DataLoader
def get_unite_dataloaders(data_path=None,train_rate=0.6,seq_len=26,pre_len=1,missing_ratio=0.6,
                    missing_index=1,batch_size=64,num_workers=8,
                    imputer=None,random=None,test=False,std=0):
    dataloaders = {}
    flows =  144
    TRAIN_X = torch.empty([0,seq_len,flows,3])
    TRAIN_Y = torch.empty([0,flows])
    VAL_X = torch.empty([0,seq_len,flows,3])
    VAL_Y = torch.empty([0,flows])
    TEST_X = torch.empty([0,seq_len,flows,3])
    TEST_Y = torch.empty([0,flows])
    for i in range(1,4):
        train_x, train_y, val_x, val_y, test_x, test_y, max_value,er = split_dataset_with_imputer(data_path, train_rate=train_rate,
                                                                            seq_len=seq_len,
                                                                            predict_len=pre_len, 
                                                                            missing_ratio=missing_ratio,missing_index=i,
                                                                            test=test,imputer=imputer,std=std
                                                                            )
        # dataloaders['er'] = er
        # dataloaders['time'] = er
        train_x = torch.from_numpy(train_x).float()
        train_y = torch.from_numpy(train_y).float()
        val_x = torch.from_numpy(val_x).float()
        val_y = torch.from_numpy(val_y).float()
        test_x = torch.from_numpy(test_x).float()
        test_y = torch.from_numpy(test_y).float()

        TRAIN_X = torch.cat((TRAIN_X,train_x), 0)
        TRAIN_Y = torch.cat((TRAIN_Y,train_y), 0)
        VAL_X = torch.cat((VAL_X,val_x), 0)
        VAL_Y = torch.cat((VAL_Y,val_y), 0)
        TEST_X = torch.cat((TEST_X,test_x), 0)
        TEST_Y = torch.cat((TEST_Y,test_y), 0)

    dataset = TensorDataset(TRAIN_X, TRAIN_Y)
    dataloaders['train'] = DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=num_workers)
    dataset = TensorDataset(VAL_X, VAL_Y)
    dataloaders['val'] = DataLoader(dataset, batch_size=batch_size, shuffle=False, pin_memory=False, num_workers=num_workers)
    dataset = TensorDataset(TEST_X, TEST_Y)
    dataloaders['test'] = DataLoader(dataset, batch_size=batch_size, shuffle=False, pin_memory=False, num_workers=num_workers)
    return dataloaders

#  get test dataset
def get_test_dataloaders(data_path=None,seq_len=26,pre_len=1,batch_size=64,num_workers=8):
    _, _, _, _, test_x, test_y, max_value = split_dataset_with_missing(data_path,seq_len=seq_len,predict_len=pre_len,
                                                                       missing_ratio=0.1,missing_index=1)
    
    pass
    
    



if __name__ == '__main__':
    # gen_missing_matrix(missing_ratio=0.0)
    dataset = 'abilene2'
    
    # gen_normal_missing_matrix(dataset='abilene',mean_ratio=0.3,std=0.05,counts=3)
    gen_normal_missing_matrix(dataset=dataset,mean_ratio=0.1,std=0.05,counts=3)
    gen_normal_missing_matrix(dataset=dataset,mean_ratio=0.2,std=0.05,counts=3)
    gen_normal_missing_matrix(dataset=dataset,mean_ratio=0.3,std=0.05,counts=3)
    gen_normal_missing_matrix(dataset=dataset,mean_ratio=0.4,std=0.05,counts=3)
    gen_normal_missing_matrix(dataset=dataset,mean_ratio=0.5,std=0.05,counts=3)
    gen_normal_missing_matrix(dataset=dataset,mean_ratio=0.6,std=0.05,counts=3)
    gen_normal_missing_matrix(dataset=dataset,mean_ratio=0.7,std=0.05,counts=3)
    gen_normal_missing_matrix(dataset=dataset,mean_ratio=0.8,std=0.05,counts=3)
    gen_normal_missing_matrix(dataset=dataset,mean_ratio=0.9,std=0.05,counts=3)
    # gen_normal_missing_matrix(dataset='abilene',mean_ratio=0.3,std=0.15,counts=3)
    # gen_normal_missing_matrix(dataset='abilene',mean_ratio=0.4,std=0.05,counts=3)
    # gen_normal_missing_matrix(dataset='abilene',mean_ratio=0.5,std=0.05,counts=3)
    # gen_normal_missing_matrix(dataset='abilene',mean_ratio=0.6,std=0.05,counts=3)
    # gen_normal_missing_matrix(dataset='abilene',mean_ratio=0.7,std=0.05,counts=3)
    # gen_missing_nodes_matrix(dataset='abilene',missing_node_num=1,counts=3)
    # gen_missing_nodes_matrix(dataset='abilene',missing_node_num=2,counts=3)
    # gen_missing_nodes_matrix(dataset='abilene',missing_node_num=3,counts=3)
    # gen_missing_nodes_matrix(dataset='abilene',missing_node_num=4,counts=3)
    pass