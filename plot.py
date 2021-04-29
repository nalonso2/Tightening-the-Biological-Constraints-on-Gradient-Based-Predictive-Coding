import pickle
import matplotlib.pyplot as plt
import torch
import numpy as np
from utilities import sigmoid_d
from utilities import tanh_d
import matplotlib
import pylab


matplotlib.rcParams['text.usetex']=False
matplotlib.rcParams['savefig.dpi']=400.
matplotlib.rcParams['font.size']=14.0
matplotlib.rcParams['figure.figsize']=(5.0,3.5)
matplotlib.rcParams['axes.formatter.limits']=[-10,10]
matplotlib.rcParams['axes.labelsize']= 14.
matplotlib.rcParams['figure.subplot.bottom'] = .2
matplotlib.rcParams['figure.subplot.left'] = .2


def compute_means(accuracies):

    mean_a = 0
    for m in range(len(accuracies)):
        mean_a += np.array(accuracies[m])
    mean_a /= len(accuracies)

    mean_a = 1 - mean_a

    return mean_a

def compute_last_mean(accuracies):

    last_accs = []
    for m in range(len(accuracies)):
        last_accs = last_accs + accuracies[m][-3:]
    mean = sum(last_accs) / len(last_accs)
    std = np.std(np.array(last_accs))

    return 1 - mean, std


def plot(fashion=False, num_iter=20):

    if fashion:
        d_name = 'fashion'
    else:
        d_name = ''


    # Plot accuracies

    with open(f'data/PC_transpose1_KP0_{num_iter}_{d_name}_sigmoid_online0_divErr0_posActs0_bias0.data', 'rb') as filehandle:
        PC = pickle.load(filehandle)
        PC_mean = compute_means(PC)

    if not fashion:
        with open(f'data/PC_transpose1_KP0_{num_iter}_{d_name}_sigmoid_online0_divErr0_posActs1_bias0.data',
                  'rb') as filehandle:
            PC_pos = pickle.load(filehandle)
            PC_pos_mean = compute_means(PC_pos)

        with open(f'data/PC_transpose1_KP0_{num_iter}_{d_name}_tanh_online0_divErr0_posActs0_bias0.data', 'rb') as filehandle:
            PC_tanh = pickle.load(filehandle)
            PC_tanh_mean = compute_means(PC_tanh)

        with open(f'data/PC_transpose1_KP0_{num_iter}_{d_name}_tanh_online0_divErr0_posActs1_bias0.data', 'rb') as filehandle:
            PC_pos_tanh = pickle.load(filehandle)
            PC_pos_tanh_mean = compute_means(PC_pos_tanh)

        with open(f'data/PC_transpose1_KP0_{num_iter}_{d_name}_tanh_online0_divErr0_posActs1_bias1.data', 'rb') as filehandle:
            PC_pos_tanh_b = pickle.load(filehandle)
            PC_pos_tanh_b_mean = compute_means(PC_pos_tanh_b)

    with open(f'data/PC_transpose1_KP0_{num_iter}_{d_name}_sigmoid_online0_divErr1_posActs1_bias1.data', 'rb') as filehandle:
        PC_div = pickle.load(filehandle)
        PC_div_mean = compute_means(PC_div)

    with open(f'data/PC_transpose0_KP1_{num_iter}_{d_name}_sigmoid_online0_divErr0_posActs0_bias0.data', 'rb') as filehandle:
        KP_PC = pickle.load(filehandle)
        KP_PC_mean = compute_means(KP_PC)

    with open(f'data/PC_transpose0_KP0_{num_iter}_{d_name}_sigmoid_online0_divErr0_posActs0_bias0.data', 'rb') as filehandle:
        Rand_PC = pickle.load(filehandle)
        Rand_PC_mean = compute_means(Rand_PC)

    with open(f'data/PC_transpose0_KP0_{num_iter}_{d_name}_sigmoid_online0_divErr1_posActs1_bias1.data', 'rb') as filehandle:
        Rand_PC_div = pickle.load(filehandle)
        Rand_PC_div_mean = compute_means(Rand_PC_div)

    with open(f'data/BP_{d_name}.data', 'rb') as filehandle:
        BP = pickle.load(filehandle)
        BP_mean = compute_means(BP)

    if not fashion:
        # Plot Accuracies for PC networks with positive activities
        #pylab.plot(BP_mean, label='Backprop', alpha=.75, linewidth=3)
        pylab.plot(PC_mean, label='PC(Sig)', alpha=.6, linewidth=3, color=(0,.8,.3))
        pylab.plot(PC_pos_mean, '--', label='PC(Sig),+Act', alpha=.95, linewidth=3, color=(0,.9,.1))
        pylab.plot(PC_tanh_mean, label='PC(Tanh)', alpha=.75, linewidth=3, color=(.4,.4,.4))
        pylab.plot(PC_pos_tanh_mean, '--', label='PC(Tanh),+Act', alpha=.75, linewidth=3, color=[.2,.2,.2])
        pylab.plot(PC_pos_tanh_b_mean, ':', label='PC(Tanh),+Act,bias', alpha=.75, linewidth=3, color=[0,0,0])
        pylab.ylim(.015, .13)
        pylab.legend()
        pylab.xlabel('Epochs')
        pylab.ylabel('Error (%)')
        pylab.tight_layout()
        pylab.savefig(f'plots/AccPosActs{d_name}.png')
        pylab.show()

    #Plot Accuracies for PC networks with different feedback weight types
    pylab.plot(BP_mean, label='Backprop', alpha=.75, linewidth=3)
    pylab.plot(PC_mean, label='PC', alpha=.75, linewidth=3)
    pylab.plot(KP_PC_mean, '--', label='KP-PC', alpha=.75, linewidth=3)
    pylab.plot(Rand_PC_mean, '--', label='Rand-PC', alpha=.75, linewidth=3)
    #pylab.title('MNIST Test Accuracy')
    #if not fashion:
        #pylab.ylim(.955, .985)
    pylab.legend()
    pylab.xlabel('Epochs')
    pylab.ylabel('Error (%)')
    pylab.tight_layout()
    pylab.savefig(f'plots/AccWeights{d_name}.png')
    pylab.show()


    # Plot Accuracies for PC Networks with different error functions
    pylab.plot(BP_mean, label='Backprop', alpha=.75, linewidth=3)
    #pylab.plot(PC_mean, label='PC w/ Sub', alpha=.75, linewidth=3)
    pylab.plot(PC_div_mean, '--', label='PC w/ Div', alpha=.75, linewidth=3)
    pylab.plot(Rand_PC_div_mean, '--', label='Rand-PC w/ Div', alpha=.75, linewidth=3)
    #pylab.title('MNIST Test Accuracy')
    pylab.legend()
    #if not fashion:
        #pylab.ylim(.93, .99)
    pylab.xlabel('Epochs')
    pylab.ylabel('Error (%)')
    pylab.tight_layout()
    pylab.savefig(f'plots/AccErr{d_name}.png')
    pylab.show()


    #Print Max Values
    print('PC Max:', max([max(PC[i]) for i in range(len(PC))]))
    print('PC-Div Max:', max([max(PC_div[i]) for i in range(len(PC_div))]))
    print('KP-PC Max:', max([max(KP_PC[i]) for i in range(len(KP_PC))]))
    print('Rand-PC Max:', max([max(Rand_PC[i]) for i in range(len(Rand_PC))]))
    print('Rand-PC-Div Max:', max([max(Rand_PC_div[i]) for i in range(len(Rand_PC))]))
    print('Backprop Max:', max([max(BP[i]) for i in range(len(BP))]))

    print('PC Mean std:', compute_last_mean(PC))
    print('PC_Div Mean std:', compute_last_mean(PC_div))
    print('KP-PC Mean std:', compute_last_mean(KP_PC))
    print('Rand_PC Mean std:', compute_last_mean(Rand_PC))
    print('Rand_PC_div Mean std:', compute_last_mean(Rand_PC_div))
    print('BackProp Mean std:', compute_last_mean(BP))


#plot()