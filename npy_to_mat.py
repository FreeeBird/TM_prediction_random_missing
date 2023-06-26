from model.LSTNet import LSTNet
from model.MTGNN import gtnet
from model.lstm import LSTM2D
from utils.data_helper import get_adj_matrix, get_data_path, get_dataloaders, get_dataset_nodes
from utils.data_process import np_to_tensor_dataset, split_dataset_with_missing
from scipy import io
import numpy as np
from utils.metrics import R2, RMSE
import torch
from torch.utils.data import DataLoader
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
            y = y.to(device)
            y_hat = model(x)
            y_true = torch.cat((y_true,y), 0)
            y_pred = torch.cat((y_pred,y_hat), 0)
            # x_true = torch.cat((x_true, x), 0)
            # x_pred = torch.cat((x_pred, x_hat), 0)
            # MASK = torch.cat((MASK, mask), 0)
    # error_ratio = ERROR_RATIO(x_pred,x_true,MASK)
    # mse = MSE(y_pred, y_true)
    rmse = RMSE(y_pred, y_true)
    r2 = R2(y_pred, y_true)
    return 0,rmse,r2


# dataset = 'geant'
seq_len = 26
train_rate = 0.6
save_path = '/data_disk/missing/'
target_ratio = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
# 获取对应数据集的测试集
# 保存到mat文件中
def save_to_mat(dataset='abilene'):
    data_path = get_data_path(dataset)
    for tr in target_ratio:
        for i in range(1,4):
            _, _, _, _, test_x, _, _=split_dataset_with_missing(
                data_path, train_rate=train_rate,seq_len=seq_len,
                predict_len=1, missing_ratio=tr,missing_index=i,
                random_train=[False,False,False],std=0.05)
            io.savemat(save_path + dataset+'_missing_'+ str(tr)+'_'+ str(i) +'.mat', {'data': test_x})

# save_to_mat('geant')
# pass

def get_pre_model(dataset='abilene',name='lstm2d'):
    if dataset=='abilene':
        flows = 144
        node =12
    else:
        flows = 529
        node = 23
    m_adj = np.load(get_adj_matrix(dataset))
    models = {
        'lstm2d':LSTM2D(flows, 64, n_layer=1,dropout=0.2),
        'lstnet':LSTNet(flows=flows,seq_len=26,pre_len=1,hidCNN=64,hidRNN=64,hidSkip=10,CNN_kernel=6,skip=2),
        'mtgnn':gtnet(gcn_true=True, buildA_true=True, gcn_depth=2, num_nodes=node, device='cuda', 
        predefined_A=m_adj, static_feat=None, 
    dropout=0.2, subgraph_size=2, node_dim=64, dilation_exponential=1, conv_channels=64, 
    residual_channels=64, 
    skip_channels=64, end_channels=64, seq_length=26, in_dim=node, out_dim=node, layers=3, 
    propalpha=0.05, tanhalpha=3, layer_norm_affline=True)
    }
    return models[name]

def get_prediction_dict(dataset='abilene',name='lstm2d'):
    abilene_dicts = {
        'lstm2d':['dict/LSTM2D_abilene_26_10-24_12:40:39_dict.pkl','dict/LSTM2D_abilene_26_10-24_13:08:28_dict.pkl','dict/LSTM2D_abilene_26_10-24_13:35:35_dict.pkl'],
        'lstnet':['dict/LSTNet_abilene_26_10-24_16:00:44_dict.pkl','dict/LSTNet_abilene_26_10-24_16:42:34_dict.pkl','dict/LSTNet_abilene_26_10-24_17:24:27_dict.pkl'],
        'mtgnn':['dict/gtnet_abilene_26_10-24_21:09:47_dict.pkl','dict/gtnet_abilene_26_10-24_22:00:45_dict.pkl','dict/gtnet_abilene_26_10-24_23:00:44_dict.pkl'],
    }
    geant_dicts = {
        'lstm2d':['dict/LSTM2D_geant_26_01-14_20:44:17_dict.pkl','dict/LSTM2D_geant_26_01-14_21:37:57_dict.pkl','dict/LSTM2D_geant_26_01-14_22:42:40_dict.pkl'],
        'lstnet':['dict/LSTNet_geant_26_02-04_15:31:43_dict.pkl', 'dict/LSTNet_geant_26_02-04_15:50:04_dict.pkl', 'dict/LSTNet_geant_26_02-04_16:09:19_dict.pkl'],
        'mtgnn':['dict/gtnet_geant_26_02-04_18:54:27_dict.pkl', 'dict/gtnet_geant_26_02-04_19:43:58_dict.pkl', 'dict/gtnet_geant_26_02-04_20:33:45_dict.pkl'],
    }
    if dataset=='abilene':
        return abilene_dicts[name]
    else:
        return geant_dicts[name]
# 读取npy文件，加载预测模型，计算填充误差RMSE，预测误差RMSE
if __name__ == '__main__':
    # save_to_mat()
    # calc_mat()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dataset = 'abilene'
    # dataset = 'geant'
    rounds = 3
    seq_len = 26
    m_adj = np.load(get_adj_matrix(dataset))
    data_path = get_data_path(dataset)
    num_nodes,num_flows = get_dataset_nodes(dataset)
    # all rounds metrices
    ALL_RMSE,ALL_R2 = [],[]
    missing_ratio = 0
    target_ratio = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
    std=0.05
    imputer_name = 'LMaFit'
    pre_model_name = 'mtgnn'
    dicts = get_prediction_dict(dataset,pre_model_name)
    model = get_pre_model(dataset,pre_model_name)
    # model = LSTM2D(529, 64, n_layer=1,dropout=0.2)
    model = model.to(device)
    mean_rmse,mean_r2=[],[]
    
    result_matrix = []
    DICT_RMSE = []
    DICT_COM_RMSE = []
    DICT_R2 = []
    DICT_TIME = []
    for tr in target_ratio:
        # if tr>0.4:
        #     break
        temp_rmse = []
        temp_com_rmse = []
        temp_r2=[]
        temp_time = []
        for i in range(1,4):
            _, _, _, _, test_x, test_y, max_value=split_dataset_with_missing(data_path, 
            train_rate=train_rate,seq_len=seq_len,predict_len=1, missing_ratio=tr,missing_index=i,
            random_train=[False,False,False],std=0.05)
            imputed_data = io.loadmat(save_path + dataset+'_'+ imputer_name+'_'+ str(tr)+'_'+ str(i) +'.mat')
            completed = imputed_data['completed_data']
            time = imputed_data['t']
            rmse2 = imputed_data['rmse']
            origin = test_x[:,:,:,0]
            com_rmse = RMSE(completed, origin)
            # test_x = np.append(test_x,)
            testset = np_to_tensor_dataset(completed, test_y)
            dataloader = DataLoader(testset, batch_size=32, 
            shuffle=False, pin_memory=False, num_workers=8)
            print("com rmse",com_rmse,rmse2)
            temp_com_rmse.append(com_rmse.item())
            temp_time.append(time)
            for d in dicts:
                model.load_state_dict(torch.load('/home/liyiyong/TM_Prediction_With_Missing_Data/'+d))
                model.eval()
                mse,rmse,r2 = test(model,dataloader,num_flows,device,seq_len)
                print(rmse,r2)
                temp_rmse.append(rmse.item())
                temp_r2.append(r2.item())
        DICT_RMSE.append(np.mean(temp_rmse))
        DICT_COM_RMSE.append(np.mean(temp_com_rmse))
        DICT_R2.append(np.mean(temp_r2))
        DICT_TIME.append(np.mean(temp_time))
        print(np.mean(temp_rmse),np.mean(temp_r2))
    print(imputer_name)
    print("ALL_Time",DICT_TIME)
    print("ALL_COM_RMSE",DICT_COM_RMSE)
    print("ALL_RMSE",DICT_RMSE)
    print("ALL_R2",DICT_R2)

    pass
