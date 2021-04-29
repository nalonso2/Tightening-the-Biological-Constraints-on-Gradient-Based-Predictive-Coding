
import PC
import torch
import torchvision
import torch.optim as optim
import utilities
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import pickle
from torch import nn
from utilities import sigmoid_d
from utilities import tanh_d
from utilities import relu_d
import numpy as np
import train_PC
import plot

def main():

    #MNIST

    #BP
    train_PC.train_network(num_models=3, bp_train=True)

    # PC-Sig
    train_PC.train_network(use_true_gradients=True, num_models=3, num_iter=20, div_err=False, pos_acts=False)

    #PC-Sig-PosActs
    train_PC.train_network(use_true_gradients=True, num_models=3, num_iter=20, div_err=False)

    # PC-Tanh
    train_PC.train_network(use_true_gradients=True, num_models=3, num_iter=20, div_err=False, f=torch.tanh, f_d=tanh_d, f_name='tanh', pos_acts=False)

    # PC-Tanh-PosActs
    train_PC.train_network(use_true_gradients=True, num_models=3, num_iter=20, div_err=False, f=torch.tanh, f_d=tanh_d, f_name='tanh')

    # PC-Tanh-PosActs-B
    train_PC.train_network(use_true_gradients=True, num_models=3, num_iter=20, div_err=False, f=torch.tanh, f_d=tanh_d, f_name='tanh', bias=3)

    #PC Division Encoding
    train_PC.train_network(use_true_gradients=True, num_models=3, num_iter=20, div_err=True, bot_infer_rate=.05, bias=1)

    #KP-PC
    train_PC.train_network(use_true_gradients=False, train_err_wts=True, decay=.01, num_models=3, num_iter=20, div_err=False, pos_acts=False)

    #Rand-PC
    train_PC.train_network(use_true_gradients=False, num_models=3, num_iter=20, div_err=False, pos_acts=False)

    #Rand-PC Division Encoding
    train_PC.train_network(use_true_gradients=False, num_models=3, num_iter=20, div_err=True, bias=1)



    # Fashion-MNIST

    # BP
    train_PC.train_network(num_models=3, bp_train=True, fashion=True)

    # PC-Sig
    train_PC.train_network(use_true_gradients=True, num_models=3, num_iter=7, top_infer_rate=.025, bot_infer_rate=.025, div_err=False, pos_acts=False, fashion=True)

    # PC Division Encoding
    train_PC.train_network(use_true_gradients=True, num_models=3, num_iter=7, div_err=True, top_infer_rate=.025, bot_infer_rate=.025, fashion=True, bias=1)

    # KP-PC
    train_PC.train_network(use_true_gradients=False, train_err_wts=True, decay=.01, top_infer_rate=.025, bot_infer_rate=.025, num_models=3, num_iter=7, div_err=False, fashion=True, pos_acts=False)

    # Rand-PC
    train_PC.train_network(use_true_gradients=False, num_models=3, num_iter=7, top_infer_rate=.025, bot_infer_rate=.025, div_err=False, fashion=True, pos_acts=False)

    # Rand-PC Division Encoding
    train_PC.train_network(use_true_gradients=False, num_models=3, num_iter=7, top_infer_rate=.025, bot_infer_rate=.025, div_err=True, fashion=True, bias=1)

    #Produce Plots
    #plot.plot()
    plot.plot(fashion=True, num_iter=7)

main()