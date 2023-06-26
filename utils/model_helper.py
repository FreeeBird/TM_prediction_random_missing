'''
Author: FreeeBird
Date: 2022-04-13 20:11:04
LastEditTime: 2022-11-08 20:28:39
LastEditors: FreeeBird
Description: 
FilePath: /TM_Prediction_With_Missing_Data/utils/model_helper.py
'''



from model.AutoEncoder import AutoEncoder
from model.LSTNet import LSTNet
from model.MTGNN import gtnet
from model.SRCNN import SRCNN
from model.autoencoder_lstm import EncoderDecoderConvLSTM
from model.conv_lstm import ConvLSTM
from model.cunet import CLUNet, UNet_LSTM
from model.gcrint import GCRINT
from model.lstm import LSTM, LSTM2D
from model.ms_unet import MSUNet_LSTM
from model.our_method import Proposed_Method
from model.transformer import Transformer_LSTM
from model.unet3d import UNet3D_LSTM, UNet3D_LSTM2D, UNet3D_LSTMF, UNet3D_LSTMT, UNet_3D
from model.unet import UNet, UNet_LSTM2D

def get_model(args):
    model_name = args.model
    model = None
    if model_name == 'Unet':
        model = UNet(1,1)
    if model_name == 'UNet_3D':
        model = UNet_3D(args.flow,1,1)
    if model_name == 'SRCNN':
        model = SRCNN()
    if model_name=='ae':
        model = AutoEncoder(26,144)
    if model_name == 'gcrint':
        model = GCRINT(args.seq_len-2,args.flow,'cuda')
    if model_name == 'LSTM':
        model = LSTM(args.flow, args.dim_model, n_layer=args.rnn_layers,dropout=args.dropout)
    if model_name == 'LSTM2D':
        model = LSTM2D(args.flow, args.dim_model, n_layer=1,dropout=args.dropout)
    if model_name == 'KNN_LSTM':
        model = LSTM(args.flow, args.dim_model, n_layer=args.rnn_layers,dropout=args.dropout)
    if model_name == 'conv_lstm':
        model = ConvLSTM(1, args.dim_model, (args.kernel_size,args.kernel_size), args.rnn_layers,
                 batch_first=True, bias=True, return_all_layers=False)
    if model_name == 'EDConv_LSTM':
        model = EncoderDecoderConvLSTM(args.dim_model,1)
    if model_name == 'Transformer_LSTM':
        model = Transformer_LSTM(args.flow,args.dim_model,args.heads,args.dropout,args.dim_ff,args.encoder_layers,args.seq_len)
    if model_name == 'UNet_LSTM':
        model = UNet_LSTM(args.in_chan,bilinear=args.bilinear,in_dim=args.flow,hidden_dim=args.dim_model,n_layer=args.rnn_layers,dropout=args.dropout)
    if model_name == 'UNet_LSTM2D':
        model = UNet_LSTM2D(args.in_chan,bilinear=args.bilinear,in_dim=args.flow,hidden_dim=args.dim_model,n_layer=args.rnn_layers,dropout=args.dropout)
    # if model_name == 'BKSUNet_LSTM':
    #     model = BKSUNet_LSTM(args.in_chan,bilinear=args.bilinear,in_dim=args.flow,hidden_dim=args.dim_model,n_layer=args.rnn_layers,dropout=args.dropout,kernel_size=args.kernel_size)
    if model_name == 'UNet3D_LSTM':
        model = UNet3D_LSTM(args.flow, args.dim_model, args.rnn_layers,args.dropout, n_channels=1, n_classes=1)
    if model_name == 'UNet3D_LSTM2D':
        model = UNet3D_LSTM2D(args.flow, args.dim_model, args.rnn_layers,args.dropout, n_channels=1, n_classes=1)
    if model_name == 'UNet3D_LSTMT':
        model = UNet3D_LSTMT(args.flow, args.dim_model, args.rnn_layers,args.dropout, n_channels=1, n_classes=1)
    if model_name == 'UNet3D_LSTMF':
        model = UNet3D_LSTMF(args.flow, args.dim_model, args.rnn_layers,args.dropout, n_channels=1, n_classes=1)
    if model_name == 'LSTNet':
        model = LSTNet(flows=args.flow,seq_len=args.seq_len,pre_len=args.pre_len,hidCNN=args.dim_model,hidRNN=args.dim_model,hidSkip=10,CNN_kernel=6,skip=2)
    if model_name == 'MTGNN':
        model = gtnet(gcn_true=True, buildA_true=True, gcn_depth=2, num_nodes=23, device=args.device, predefined_A=args.m_adj, static_feat=None, 
    dropout=args.dropout, subgraph_size=2, node_dim=args.dim_model, dilation_exponential=1, conv_channels=args.dim_model, residual_channels=args.dim_model, 
    skip_channels=args.dim_model, end_channels=args.dim_model, seq_length=args.seq_len, in_dim=23, out_dim=23, layers=3, propalpha=0.05, tanhalpha=3, layer_norm_affline=True)
    if model_name == 'PM':
        model = Proposed_Method(mc_module=args.mc_mod,mp_module=args.mp_mod, n_channels=1, bilinear=True,in_dim=args.flow, hidden_dim=args.flow,
                            n_layer=args.rnn_layers,dropout=args.dropout,timestep=args.seq_len,args=args)
        # (1, 1, bilinear=False,in_dim=args.flow, hidden_dim=args.dim_model, n_layer=args.rnn_layers,dropout=args.dropout)
    return model.to(args.device)