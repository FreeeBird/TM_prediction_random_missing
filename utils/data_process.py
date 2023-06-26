import os
import sys

import tqdm

from model.HaLRTC import HaLRTC

# from model.unet3d import UNet3D_LSTM2D
sys.path.append(os.getcwd())
import numpy as np
from sklearn.impute import KNNImputer
import torch
from scipy.stats import truncnorm
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
from sklearn import preprocessing

def min_max_normalization(data, max_data=None, min_data=None):
    if max_data is None:
        max_data = np.max(data)
        min_data = np.min(data)
    _range = max_data - min_data
    return (data - min_data) / _range


def z_score(data):
    return (data - np.mean(data)) / np.std(data)


def col_max_norm(data, col_max=None, col_min=None):
    if col_max is None:
        col_max = np.max(data, axis=0)
        col_min = np.min(data, axis=0)
    # data = (data - col_min) / (col_max - col_min)
    data = data  / col_max
    data[np.isnan(data)] = 0.0
    data[np.isinf(data)] = 0.0
    return data


def MinMaxScale(data):
    scaler = preprocessing.MinMaxScaler()
    scaled_data = scaler.fit_transform(data)
    return scaled_data

def max_normalization(data):
    return data / np.max(data)


def inverse_normalization(data, col_max, col_min):
    return data * (col_max - col_min) + col_min


def mean_normalization(data):
    return (data - np.mean(data)) / (np.max(data) - np.min(data))


def normalization(data, norm_type='min_max_norm'):
    if norm_type == 'min_max_norm':
        data = min_max_normalization(data)
    if norm_type == 'z_score':
        data = z_score(data)
    if norm_type == 'max_norm':
        data = max_normalization(data)
    if norm_type == 'mean_norm':
        data = mean_normalization(data)
    return data

# given a mask ratio
# Mask Generation
# for every trffaic matrix e.g. len=26
# 1 means missing 
def random_mask_2d(row=12, col=144, mask_rate=0.5):
    mask = np.zeros((row,col))
    for i in range(row):
        tmp = np.zeros(col)
        tmp[:int(col * mask_rate)] = 1
        np.random.shuffle(tmp)
        mask[i] = tmp
    return mask


def uniform_mask_2d(row=12, col=144, mask_rate=0.5):
    len_keep = int(col * (1 - mask_rate))
    noise = np.random.rand(row,col)  # noise in [0, 1]
    ids_shuffle = np.argsort(noise,axis=1)# ascend: small is keep, large is remove
    ids_restore = np.argsort(ids_shuffle, axis=1)
    # keep the first subset
    # ids_keep = ids_shuffle[:, :len_keep]
    # generate the binary mask: 0 is keep, 1 is remove
    mask = np.ones((row,col))
    mask[:, :len_keep] = 0
    # unshuffle to get the binary mask
    mask = np.take_along_axis(mask,indices=ids_restore,axis=1)
    return mask

def uniform_masking(data,mask_ratio=0.6):
    N, L = data.shape  # batch, length
    len_keep = int(L * (1 - mask_ratio))
    noise = np.random.rand(N,L)  # noise in [0, 1]
    ids_shuffle = np.argsort(noise,axis=1)# ascend: small is keep, large is remove
    ids_restore = np.argsort(ids_shuffle, axis=1)
    # keep the first subset
    # ids_keep = ids_shuffle[:, :len_keep]
    # generate the binary mask: 0 is keep, 1 is remove
    mask = np.ones((N,L))
    mask[:, :len_keep] = 0
    # unshuffle to get the binary mask
    mask = np.take_along_axis(mask,indices=ids_restore,axis=1)
    data = np.stack([data,mask],axis=-1)
    return data

""" 
   using Slide Window Method to produce traffic matrix sequence with seq_len with missing values
   
"""
def matrix_sample_maker(data, seq_len=12, pre_len=1, missing_ratio=0.5,missings=[],random=False):
    x, y = [], []
    time_len = data.shape[0]
    if random:
        mask_rates = np.random.normal(loc = 0.3 , scale = 0.05,size=time_len - seq_len - pre_len)
        import matplotlib.pyplot as plt
        print("最大值：",np.max(mask_rates))
        print("最小值：",np.min(mask_rates))
        print("平均值：",np.mean(mask_rates))
        print("中值：",np.median(mask_rates))
        print("均方差：",np.std(mask_rates))
        plt.hist(mask_rates,range=(0,1),density=True)
        plt.savefig('fig2.png')
    for i in range(time_len - seq_len - pre_len):
        sample = data[i:i + seq_len + pre_len]
        # 随机生成mask矩阵
        # mask = random_mask_2d(seq_len, data.shape[1], missing_ratio)
        # 均匀分布
        if random:
            # miss = uniform_mask_2d(seq_len, data.shape[1], int(mask_rates[i]))
            miss = random_mask_2d(seq_len, data.shape[1], mask_rates[i])
        else:
            miss = missings[i:i+seq_len]
        tmp_x = np.dstack((sample[0:seq_len], miss))
        x.append(tmp_x)
        y.extend(sample[seq_len:seq_len + pre_len])
    x, y = np.array(x), np.array(y)
    return x, y


""" 
    Not Used Method
"""
def flow_sample_maker(data, seq_len=3, pre_len=1,missings=[]):
    # assert data = [time,flows]
    # data = [flows,time]
    data = data.T
    missings = missings.T
    x, y = [], []
    time_len = data.shape[1]
    for i in range(time_len - seq_len - pre_len):
        sample = data[:, i:i + seq_len + pre_len]
        miss = missings[:,i:i+seq_len]
        tmp_x = np.dstack((sample[:, 0:seq_len], miss))
        x.extend(tmp_x)
        y.extend(sample[:, seq_len:seq_len + pre_len])
    x, y = np.array(x), np.array(y)
    return x, y

""" 
   using Slide Window Method to produce traffic matrix sequence with seq_len without missing values
"""
def matrix_sample_maker_with_no_mask(data, seq_len=12, pre_len=1):
    x, y = [], []
    time_len = data.shape[0]
    for i in range(time_len - seq_len - pre_len):
        sample = data[i:i + seq_len + pre_len]
        x.append(sample[:seq_len])
        y.extend(sample[seq_len:seq_len + pre_len])
    x, y = np.array(x), np.array(y)
    return x, y

""" 
    1.read traffic data from data_path into ndarray data
    2.using Max Normalization to tranform data into 0~1
    3.split dataset with train_ratio:val_ratio:test_ratio=6:2:2
"""
def split_dataset(data_path, train_rate=0.6, test_rate=0.2, seq_len=12, predict_len=1):
    data = np.load(data_path) / 1000.0
    # norm-way
    if data_path.find('abilene')>0:
        print('max-norm')
        data = normalization(data, "max_norm")
    else:
        print('col-norm')
        data = col_max_norm(data)
    # data = MinMaxScale(data)
    max_value = np.max(data)
    time_len = data.shape[0]
    val_index = int(time_len * train_rate)
    test_index = int(time_len * (1 - test_rate))
    # val_index = int(time_len*(1-test_rate))
    test_data = data[test_index:]
    train_data = data[:val_index]
    val_data = data[val_index:test_index]
    # 在各个部分分别生成样本
    x_test, y_test = matrix_sample_maker_with_no_mask(test_data, seq_len, predict_len)
    x_train, y_train = matrix_sample_maker_with_no_mask(train_data, seq_len, predict_len)
    x_val, y_val = matrix_sample_maker_with_no_mask(val_data, seq_len, predict_len)
    # x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=train_rate)
    # x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, train_size=(train_rate/(1-test_rate)))
    return x_train, y_train, x_val, y_val, x_test, y_test, max_value


""" 
    1.read traffic data from data_path into ndarray data
    2.using Max Normalization to tranform data into 0~1
    3.read mask matrix data with missing_datio and std params frpm npy file
    4.split dataset with train_ratio:val_ratio:test_ratio=6:2:2
"""
def split_dataset_with_missing(data_path, train_rate=0.6, test_rate=0.2, seq_len=12, predict_len=1, missing_ratio=0.5,std=0,missing_index=1,
random_train=[False]*3):
    data = np.load(data_path) / 1000.0
    # norm-way
    if data_path.find('abilene')>0:
        print('max-norm')
        data = normalization(data, "max_norm")
    else:
        print('col-norm')
        data = col_max_norm(data)
    # data = col_max_norm(data)
    
    if std>0:
        dataset_path = data_path.split(".")[0] + "_missing_" + str(missing_ratio)+"+-"+str(std) + "_" + str(missing_index) + ".npy"
    else:
        dataset_path = data_path.split(".")[0] + "_missing_" + str(missing_ratio)+ "_" + str(missing_index) + ".npy"
    print('DatasetPath:',dataset_path)
    missing_matrix = np.load(dataset_path)
    max_value = np.max(data)
    time_len = data.shape[0]
    val_index = int(time_len * train_rate)
    test_index = int(time_len * (1 - test_rate))
    # val_index = int(time_len*(1-test_rate))
    test_data = data[test_index:]
    train_data = data[:val_index]
    val_data = data[val_index:test_index]
    # 在各个部分分别生成样本
    x_test, y_test = matrix_sample_maker(test_data, seq_len, predict_len, missing_ratio,missings=missing_matrix[test_index:],random=False)
    x_train, y_train = matrix_sample_maker(train_data, seq_len, predict_len, 0,missings=missing_matrix[:val_index],random=False)
    x_val, y_val = matrix_sample_maker(val_data, seq_len, predict_len, missing_ratio,missings=missing_matrix[val_index:test_index],random=False)
    # x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=train_rate)
    # x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, train_size=(train_rate/(1-test_rate)))
    return x_train, y_train, x_val, y_val, x_test, y_test, max_value


def split_dataset_with_imputer(data_path, train_rate=0.6, test_rate=0.2, seq_len=12, predict_len=1, 
                          missing_ratio=0.6,std=0.05,missing_index=1,test=False,imputer='knn'):
    data = np.load(data_path) / 1000.0
    # norm-way
    if data_path.find('abilene')>0:
        print('max-norm')
        data = normalization(data, "max_norm")
    else:
        print('col-norm')
        data = col_max_norm(data)
    if std>0:
        dataset_path = data_path.split(".")[0] + "_missing_" + str(missing_ratio)+"+-"+str(std) + "_" + str(missing_index) + ".npy"
    else:
        dataset_path = data_path.split(".")[0] + "_missing_" + str(missing_ratio)+ "_" + str(missing_index) + ".npy"
    print('DatasetPath:',dataset_path)
    missing_matrix = np.load(dataset_path)
    # missing_matrix = np.load(data_path.split(".")[0] + "_missing_" + str(missing_ratio) + "_"+ str(missing_index) + ".npy")
    max_value = np.max(data)
    time_len = data.shape[0]
    val_index = int(time_len * train_rate)
    test_index = int(time_len * (1 - test_rate))
    # val_index = int(time_len*(1-test_rate))
    test_data = data[test_index:]
    train_data = data[:val_index]
    val_data = data[val_index:test_index]
    # 在各个部分分别生成样本
    x_test, y_test = matrix_sample_maker(test_data, seq_len, predict_len, missing_ratio,missings=missing_matrix[test_index:])
    x_train, y_train = matrix_sample_maker(train_data, seq_len, predict_len, missing_ratio,missings=missing_matrix[:val_index],random=False)
    x_val, y_val = matrix_sample_maker(val_data, seq_len, predict_len, missing_ratio,missings=missing_matrix[val_index:test_index])
    imputer_fun = get_imputer(imputer)
    if not test:
        x_train = imputer_fun(x_train)
        x_val = imputer_fun(x_val)
    import time
    time_start = time.time() 
    x_test = imputer_fun(x_test)
    time_end = time.time()  # 记录结束时间
    time_sum = time_end - time_start  # 计算的时间差为程序的执行时间，单位为秒/s
    print(time_sum)
    # x_test = normalization(x_test, "max_norm")
    # t_max = np.max(x_test)
    # x_test,y_test = x_test/t_max,y_test/t_max
    # x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=train_rate)
    # x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, train_size=(train_rate/(1-test_rate)))
    return x_train, y_train, x_val, y_val, x_test, y_test, max_value,time_sum


def get_imputer(ip='knn'):
    if ip=='knn':
        return knn_imputer
    if ip=='mean':
        return mean_imputer
    if ip=='all_mean':
        return all_mean_imputer
    if ip=='matrix_mean':
        return matrix_mean_imputer
    if ip=='u3d':
        return u3d_imputer
    if ip=='si':
        return softImputer
    if ip=='bci':
        return BisCalerImputer
    if ip=='nnmi':
        return NuclearNormMinimizationImputer
    if ip=='halrtc':
        return HaLRTC_imputer
    if ip=='one':
        return one_imputer


def all_mean_imputer(data):
    n,t,f,_ = data.shape
    mask = data[:,:,:,1]
    data = data[:,:,:,0]
    x_pred = np.zeros([n,t,f,3])
    for i in tqdm.trange(n):
        x_pred[i,:,:,0] = data[i,:,:].copy()
        x = data[i,:,:]
        m = mask[i,:,:]
        x = np.where(m>0, np.nan, x)
        mean = np.nanmean(x)
        x[np.isnan(x)] = mean
        # x = x + mean*(m) 
        x_pred[i,:,:,2] = x
        x_pred[i,:,:,1] = m
        
    return x_pred

def matrix_mean_imputer(data):
    n,t,f,_ = data.shape
    mask = data[:,:,:,1]
    data = data[:,:,:,0]
    x_pred = np.zeros([n,t,f,3])
    for i in tqdm.trange(n):
        x_pred[i,:,:,0] = data[i,:,:].copy()
        x = data[i,:,:]
        m = mask[i,:,:]
        x = np.where(m>0, np.nan, x)
        for j in range(t):
            mean = np.nanmean(x[j])
            x[j,np.isnan(x[j])] = mean
        # x = x + mean*(m) 
        x_pred[i,:,:,2] = x
        x_pred[i,:,:,1] = m
        
    return x_pred

def one_imputer(data):
    n,t,f,_ = data.shape
    mask = data[:,:,:,1]
    data = data[:,:,:,0]
    x_pred = np.zeros([n,t,f,3])
    for i in tqdm.trange(n):
        x_pred[i,:,:,0] = data[i,:,:].copy()
        x = data[i,:,:]
        m = mask[i,:,:]
        x = np.where(m>0, np.nan, x)
        # mean = np.nanmean(x)
        x[np.isnan(x)] = 1
        # x = x + mean*(m) 
        x_pred[i,:,:,2] = x
        x_pred[i,:,:,1] = m
        
    return x_pred

def HaLRTC_imputer(data):
    # todo 
    n,t,f,_ = data.shape
    node = 12
    mask = data[:,:,:,1]
    data = data[:,:,:,0]
    x_pred = np.zeros([n,t,f,3])
    for i in tqdm.trange(n):
        x_pred[i,:,:,0] = data[i,:,:].copy()
        x = np.expand_dims(data[i,:,:],axis=0)
        m = np.expand_dims(mask[i,:,:],axis=0)
        # x = data[i,:,:].reshape([t,node,node])
        # m = mask[i,:,:].reshape([t,node,node])
        x_hat = HaLRTC(X=x*(1-m),Z=(1-m))
        x_pred[i,:,:,2] = x_hat.squeeze()
        x_pred[i,:,:,1] = m.squeeze()
    return x_pred

def u3d_imputer(data):
    n,t,f,_ = data.shape
    mask = data[:,:,:,1]
    # data = data[:,:,:,0]
    # model = UNet3D_LSTM2D(144, 64, 3,0.2, n_channels=1, n_classes=1)
    # model.load_state_dict(torch.load('dict/UNet3D_LSTM2D_abilene_26_10-31_21:57:02_dict.pkl'))
    # model.to('cuda')
    # model.eval()
    # pred = model.restruction(data,mask)
    # return pred

def mean_imputer(data):
    n,t,f,_ = data.shape
    mask = data[:,:,:,1]
    data = data[:,:,:,0]
    x_pred = np.zeros([n,t,f,3])
    for i in tqdm.trange(n):
        x = data[i,:,:]
        x_pred[i,:,:,0] = x.copy()
        m = mask[i,:,:]
        a_mean = np.nanmean(x)
        x = np.where(m>0, np.nan, x)
        # for j in range(f):
        #     mean = np.nanmean(x[:,j])
        #     if np.isnan(mean):
        #        mean = a_mean*0
        #     x[:,j][np.isnan(x[:,j])] = mean
        mean = np.tile(np.nanmean(x,axis=0),(t,1))
        # if(np.sum(np.isnan(mean))):
        #     print(np.nanmean(x,axis=0))
        
        mean[np.isnan(mean)] = 0
        # mean[np.isnan(mean)] = a_mean
        # mean = np.nanmin(x)
        x[np.isnan(x)] = 0
        # print('x',x)
        # print('mean',mean)
        # print('m',m)
        x = x + mean*m
        # print(x)
        x_pred[i,:,:,2] = x
        x_pred[i,:,:,1] = m
    return x_pred
    
from fancyimpute import NuclearNormMinimization, SoftImpute, BiScaler
def softImputer(data):
    n,t,f,_ = data.shape
    mask = data[:,:,:,1]
    data = data[:,:,:,0]
    x_pred = np.zeros([n,t,f,3])
    x_pred[:,:,:,0] = data.copy()
    x_pred[:,:,:,1] = mask
    # data = np.where(mask>0, np.nan, data)
    # print(data.shape)
    # x_pred[:,:,:,2] = SoftImpute(verbose=False,min_value=0).fit(data)
    # SoftImpute(verbose=False,convergence_threshold=0.001,
    # max_value=1,min_value=0,max_iters=100).fit_transform(data)
    # for i in tqdm.trange(n):
    for i in range(n):
        x = data[i,:,:]
        x_pred[i,:,:,0] = x.copy()
        m = mask[i,:,:]
        x_pred[i,:,:,1] = m
        x = np.where(m>0, np.nan, x)
        x_pred[i,:,:,2] = SoftImpute(verbose=False,convergence_threshold=0.0001,
        max_value=1,min_value=0,max_iters=100,max_rank=t,init_fill_method='mean').fit_transform(x)
        # x[np.isnan(x)] = 0
        # x_pred[i,:,:,2] = x
    return x_pred

def BisCalerImputer(data):
    n,t,f,_ = data.shape
    mask = data[:,:,:,1]
    data = data[:,:,:,0]
    x_pred = np.zeros([n,t,f,3])
    for i in tqdm.trange(n):
        x = data[i,:,:]
        x_pred[i,:,:,0] = x.copy()
        m = mask[i,:,:]
        x_pred[i,:,:,1] = m
        x = np.where(m>0, np.nan, x)
        # a_mean = np.nanmean(x)
        x_pred[i,:,:,2] = BiScaler(verbose=False).fit_transform(x)
        # x[np.isnan(x)] = 0
        # x_pred[i,:,:,2] = x
    return x_pred

def NuclearNormMinimizationImputer(data):
    from fancyimpute import MatrixFactorization
    n,t,f,_ = data.shape
    mask = data[:,:,:,1]
    data = data[:,:,:,0]
    x_pred = np.zeros([n,t,f,3])
    for i in tqdm.trange(n):
        x = data[i,:,:]
        x_pred[i,:,:,0] = x.copy()
        m = mask[i,:,:]
        x_pred[i,:,:,1] = m
        x = np.where(m>0, np.nan, x)
        # a_mean = np.nanmean(x)
        x_pred[i,:,:,2] = MatrixFactorization(min_value=0,verbose=False).fit_transform(x)
        # x[np.isnan(x)] = 0
        # x_pred[i,:,:,2] = x
    return x_pred

def knn_imputer(data,n_neighbors=2):
    # n,t,f,_ = data.shape
    # x_true, x_pred = np.zeros([n,t,f]),np.zeros([n,t,f])
    # MASK = np.zeros([n,t,f])
    # for i in tqdm.trange(n):
    #     x = data[i,:,:,0]
    #     mask = data[i,:,:,1]
    #     x_true[i] = x
    #     MASK[i] = mask
    #     x = np.where(mask>0, np.nan, x)
    #     imputer = KNNImputer(n_neighbors=n_neighbors)
    #     d_m = np.count_nonzero(np.isnan(x),axis=0)
    #     d_i = np.where(d_m>(t-1),True,False)
    #     x[:,d_i] = 0.
    #     data[i,:,:,0] = imputer.fit_transform(x)
    #     x_pred[i] = data[i,:,:,0]
        # if i == 10:
        #     break
    # error_ratio = torch.sqrt(torch.sum(((x_pred - x_true) * mask) ** 2.0)) / torch.sqrt(torch.sum(target * mask))
    # error_ratio = np.sqrt(np.sum(((x_pred - x_true) * MASK) ** 2.0)) / np.sqrt(np.sum((x_true * MASK)**2.0))
    # print(error_ratio)
    # return data[:,:,:,0]
    from fancyimpute import KNN
    n,t,f,_ = data.shape
    x_pred = np.zeros([n,t,f,3])
    for i in tqdm.trange(n):
        x = data[i,:,:,0].copy()
        x_pred[i,:,:,0] = x.copy()
        mask = data[i,:,:,1]
        x_pred[i,:,:,1] = mask
        x = np.where(mask>0, np.nan, x)
        # imputer = KNNImputer(n_neighbors=n_neighbors)
        # d_m = np.count_nonzero(np.isnan(x),axis=0)
        # d_i = np.where(d_m>(t-1),True,False)
        # x[:,d_i] = 0.
        # data[i,:,:,0] = imputer.fit_transform(x)
        x_pred[i,:,:,2] = KNN(k=2,verbose=False,min_value=0,max_value=1).fit_transform(x)
        #     break
    # error_ratio = torch.sqrt(torch.sum(((x_pred - x_true) * mask) ** 2.0)) / torch.sqrt(torch.sum(target * mask))
    # error_ratio = np.sqrt(np.sum(((x_pred - x_true) * MASK) ** 2.0)) / np.sqrt(np.sum((x_true * MASK)**2.0))
    # print(error_ratio)
    return x_pred

#  Not Used Method
def random_split_dataset(data_path, train_rate=0.8, test_rate=0.2, seq_len=12, predict_len=1, mask_rate=0.5):
    data = np.load(data_path) / 1000.0
    data = normalization(data, "max_norm")
    max_value = np.max(data)
    # 在各个部分分别生成样本
    x, y = matrix_sample_maker(data, seq_len, predict_len, mask_rate)
    # array -> np array
    x, y = np.array(x), np.array(y)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_rate)
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, train_size=(train_rate / (1 - test_rate)))
    return x_train, y_train, x_val, y_val, x_test, y_test, max_value

#  ndarray/array -> TensorDataset
def np_to_tensor_dataset(x, y):
    x, y = torch.from_numpy(x).float(), torch.from_numpy(y).float()
    return TensorDataset(x, y)

# TensorDataset->dataloader
def get_dataloader(x, y, bs=32, shuffle=True, pip_memory=True, num_workers=8):
    dataset = np_to_tensor_dataset(x, y)
    dl = DataLoader(dataset, batch_size=bs, shuffle=shuffle, pin_memory=pip_memory, num_workers=num_workers)
    return dl

# Not Used Method
def split_dataset_without_sample(data_path, train_rate=0.6, test_rate=0.2, mask_rate=0.6):
    data = np.load(data_path)
    data = col_max_norm(data)
    max_value = np.max(data)
    time_len = data.shape[0]
    val_index = int(time_len * train_rate)
    test_index = int(time_len * (1 - test_rate))
    # val_index = int(time_len*(1-test_rate))
    test_data = uniform_masking(data[test_index:])
    train_data = uniform_masking(data[:val_index])
    val_data = uniform_masking(data[val_index:test_index])
    return train_data, val_data, test_data, max_value


# for i in range(1,4):
#     data_path = get_data_path('abilene')
#     split_dataset_imputer(data_path, train_rate=0.6, test_rate=0.2, seq_len=12, predict_len=1, missing_ratio=0.75,missing_index=i,imputation='knn')
#     split_dataset_imputer(data_path, train_rate=0.6, test_rate=0.2, seq_len=12, predict_len=1, missing_ratio=0.9,missing_index=i,imputation='knn')

#  Generate a set of number with normal manner in 0~1
def get_truncated_normal(mean=0, sd=1, low=0, upp=10):
    return truncnorm(
        (low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd)


#  
if __name__ == '__main__':
    # mask_rates = np.random.normal(loc = 0 , scale = 1,size=10000)
    # import matplotlib.pyplot as plt
    from fancyimpute import KNN, NuclearNormMinimization, SoftImpute, BiScaler
    t1=np.random.random((3,4)).astype(float)
    
    m1=np.random.random((3,4))
    m1 = np.where(m1>0.5,1,0)
    print(t1)
    print(m1)
    # ip = np.stack([t1,m1],axis=3)
    # print(ip,ip.shape)
    X_incomplete_normalized = np.where(m1>0,np.nan,t1)
    print(X_incomplete_normalized)
    X_filled_softimpute = SoftImpute().fit_transform(X_incomplete_normalized)
    softImpute_mse = ((X_filled_softimpute[m1] - t1[m1]) ** 2).mean()
    print("SoftImpute MSE: %f" % softImpute_mse)
    # from scipy.stats import truncnorm
    # gener= get_truncated_normal(0.2,0.05,low=0)
    # mask_rates = gener.rvs(10000)
    # print("最大值：",np.max(mask_rates))
    # print("最小值：",np.min(mask_rates))
    # print("平均值：",np.mean(mask_rates))
    # print("中值：",np.median(mask_rates))
    # print("标准差：",np.std(mask_rates))
    # print("方差",np.var(mask_rates))
    # plt.hist(mask_rates,range=(0,1),density=True,bins=100)
    # plt.savefig('fig2.png')
    
    pass
