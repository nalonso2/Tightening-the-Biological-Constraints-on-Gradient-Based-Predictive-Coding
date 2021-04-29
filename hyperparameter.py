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



def hyperparam_search(div_err=True, data_fashion=True, true_grads=True):

    top_infer_rate = [.1]
    bot_infer_rate = [.01, .025, .05, .075, .1]
    num_iter = [3, 5, 10, 15, 20]

    max_tot = 0
    max_end = 0

    best_tot = []
    best_end = []

    count = 0
    for ti in top_infer_rate:
        for bi in bot_infer_rate:
            for nit in num_iter:
                count += 1
                print('Percent Complete:', count/(len(top_infer_rate) * len(bot_infer_rate) * len(num_iter)))

                m_accs = train_PC.train_network(use_true_gradients=true_grads, num_models=1, num_iter=nit, div_err=div_err,
                                       bot_infer_rate=bi, top_infer_rate=ti, fashion=data_fashion, record=False)

                if max(m_accs[0]) > max_tot:
                    best_tot = [ti, bi, nit]
                    max_tot = max(m_accs[0])

                if m_accs[0][-1] > max_end:
                    best_end = [ti, bi, nit]
                    max_end = m_accs[0][-1]

    print('Model: Div_error?', div_err, 'RealGrads?', true_grads, 'FashionData?', data_fashion)

    print('Best Last Accuracy:', max_end, 'TopInferRate:', best_end[0],'BotInferRate:', best_end[1],
          'num iterations:', best_end[2])

    print('Best Overall Accuracy:', max_tot, 'TopInferRate:', best_tot[0], 'BotInferRate:', best_tot[1],
          'num iterations:', best_tot[2])


#Div error with true gradients
#print('TEST 1: Div Error w/ True Gradients')
#hyperparam_search()

#Div err with random feedback
#print('TEST 2: Div Error w/ Random Feedback')
#hyperparam_search(true_grads=False)

#Subtract error with true grads
print('TEST 3: Subtract Error w/ True Gradients')
hyperparam_search(div_err=False)