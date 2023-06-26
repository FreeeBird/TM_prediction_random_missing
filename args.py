'''
Author: FreeeBird
Date: 2022-04-13 16:32:06
LastEditTime: 2022-11-03 15:58:24
LastEditors: FreeeBird
Description: 
FilePath: /TM_Prediction_With_Missing_Data/args.py
'''
import argparse
from configs import config

def parse_args():
    """
    Parse command line arguments.

    Args:
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default=config['model'], help='model name')
    parser.add_argument('--mp_mod', default=config['mp_mod'], help='mp_mod name')
    parser.add_argument('--mc_mod', default=config['mc_mod'], help='mc_mod name')
    parser.add_argument('--dataset', default=config['dataset'], help='chose dataset', choices=['geant', 'abilene'])
    parser.add_argument('--gpu', default=config['gpu'],type=int, help='use -1/0/1 chose cpu/gpu:0/gpu:1', choices=[-1, 0, 1])
    parser.add_argument('--device', default=config['device'],type=int, help='device')
    parser.add_argument('--cpu', default=config['cpu'],type=int, help='cpu cores')
    parser.add_argument('--epochs', default=config['epochs'],type=int, help='epochs')
    parser.add_argument('--batch_size', '--bs', default=config["batch_size"],type=int, help='batch_size')
    parser.add_argument('--learning_rate', '--lr', default=config["learning_rate"], help='learning_rate')
    parser.add_argument('--seq_len', default=config["seq_len"], type=int, help='input history length')
    parser.add_argument('--pre_len', default=config["pre_len"], type=int, help='prediction length')
    parser.add_argument('--flow', default=config["flow"], type=int, help='flows')
    parser.add_argument('--heads', default=config["heads"], type=int, help='heads')
    parser.add_argument('--dim_ff', default=config["dim_ff"], type=int, help='dim_ff')
    parser.add_argument('--dim_model', default=config["dim_model"], help='dimension of embedding vector')
    parser.add_argument('--train_rate', default=config["train_rate"], help='')
    parser.add_argument('--test_rate', default=config["test_rate"], help='')
    parser.add_argument('--rnn_layers', default=config["rnn_layers"], help='rnn layers')
    parser.add_argument('--encoder_layers', default=config["encoder_layers"], help='encoder layers')
    parser.add_argument('--dropout', default=config["dropout"], help='dropout rate')
    parser.add_argument('--std', default=config["std"], help='std')
    parser.add_argument('--lw', default=config["lw"], help='loss1 weight')
    parser.add_argument('--bilinear', default=config["bilinear"], help='bilinear')
    parser.add_argument('--in_chan', default=config["in_chan"], help='dropout rate')
    parser.add_argument('--kernel_size', default=config["kernel_size"], help='kernel_size rate')
    parser.add_argument('--missing_ratio','--ms', default=config["missing_ratio"],type=float, help='missing rate')
    parser.add_argument('--early_stop','--es',type=int, default=config["early_stop"], help='early stop patient epochs')
    parser.add_argument('--rounds', default=config["rounds"], help='rounds')
    parser.add_argument("--do-train", default=True, type=lambda x: (str(x).lower() == "true"),
                        help="whether or not to train the model")
    parser.add_argument("--do-eval", default=False, type=lambda x: (str(x).lower() == "true"),
                        help="whether or not evaluating the mode")
    parser.add_argument("--output-dir", default=config["output_dir"], help="name of folder to output files")
    parser.add_argument("--log-dir", default=config["log_dir"], help="name of folder to log files")
    parser.add_argument("--tensorboard-dir", default=config["tensorboard_dir"], help="name of folder to tensorboard_dir files")
    parser.add_argument("--ckpt", default=None, help="checkpoint path for evaluation")
    return parser.parse_args()
