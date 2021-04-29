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

mse = torch.nn.MSELoss(reduction='none')

################ HELPER FUNCTIONS #######################################

# Load MNIST Data
def get_data(batch_size=64, fashion=False):

    if fashion:
        d_name = 'fashion'

        train_loader = DataLoader(
            torchvision.datasets.FashionMNIST('/files/', train=True, download=True,
                                       transform=torchvision.transforms.Compose([
                                           torchvision.transforms.ToTensor(),
                                       ])), batch_size=batch_size, shuffle=True, pin_memory=False)

        test_loader = DataLoader(
            torchvision.datasets.FashionMNIST('/files/', train=False, download=True,
                                       transform=torchvision.transforms.Compose([
                                           torchvision.transforms.ToTensor(),
                                       ])), batch_size=10000, shuffle=True, pin_memory=False)
    else:
        d_name = ''

        train_loader = DataLoader(
            torchvision.datasets.MNIST('/files/', train=True, download=True,
                                              transform=torchvision.transforms.Compose([
                                                  torchvision.transforms.ToTensor(),
                                              ])), batch_size=batch_size, shuffle=True, pin_memory=False)

        test_loader = DataLoader(
            torchvision.datasets.MNIST('/files/', train=False, download=True,
                                              transform=torchvision.transforms.Compose([
                                                  torchvision.transforms.ToTensor(),
                                              ])), batch_size=10000, shuffle=True, pin_memory=False)

    return train_loader, test_loader, d_name



def compute_means(models, accuracies):

    mean_a = 0
    for m in range(len(models)):
        mean_a += np.array(accuracies[m])
    mean_a /= len(models)

    return mean_a



def compute_stds(means):
    return




############### TRAIN FUNCTION ##################################


def train_network(n_epochs=30, batch_size=64, num_iter=1, f_l_rate=.001, train_online=False, decay=0.0, bot_infer_rate=.1,
                  top_infer_rate=.1, use_true_gradients=True, f=torch.sigmoid, f_d=sigmoid_d, f_name='sigmoid', pos_acts=True,
                  num_models=1, fashion=False, train_err_wts=False, div_err=False, bp_train=False, record=True, bias=0):

    ###########################################################

    #Get training and testing data
    train_loader, test_loader, d_name = get_data(batch_size, fashion)

    ##########################################################

    # Create Models
    models = []
    for m in range(num_models):
        models.append(PC.PC([784, 300, 300, 10], f_l_rate=f_l_rate, n_iter=num_iter, weight_decay=decay, pos_acts=pos_acts,
                            online=train_online, bot_infer_rate=bot_infer_rate, top_infer_rate=top_infer_rate,
                            true_gradients=use_true_gradients, func=f, func_d=f_d, train_err_wts=train_err_wts,
                            div_err=div_err, bp_train=bp_train, bias=bias))


    ###########################################################

    # Create data containers
    models_accuracies = [[] for x in range(num_models)]

    # Create Containers
    activities = [torch.zeros(1, 1) for i in range(models[0].num_layers)]
    errors = [torch.zeros(1, 1) for i in range(models[0].num_layers)]
    predictions = [torch.zeros(1, 1) for i in range(models[0].num_layers - 1)]


    ###########################################################
    #Train and test

    for ep in range(n_epochs):

        # TRAIN
        for batch_idx, (images, y) in enumerate(train_loader):
            images = images.view(-1, 784)

            # Transform targets, y, to onehot vector
            target = torch.zeros(images.size(0), 10)
            target = target
            utilities.to_one_hot(target, y)

            if models[m].bp_train:
                # Train each network
                for m in range(num_models):
                    models[m].train_bp(images, target)
            else:
                #Train each network
                with torch.no_grad():
                    for m in range(num_models):
                        models[m].train_network(images, activities, predictions, errors, target)


        # TEST Accuracy, Gradient Similarity, and alignment
        with torch.no_grad():
            for batch_idx, (images, y) in enumerate(test_loader):
                images = images.view(-1, 784)

                # Transform targets, y, to onehot vector
                target = torch.zeros(images.size(0), 10)
                utilities.to_one_hot(target, y)

                for m in range(num_models):

                    models[m].initialize_values(images, activities, predictions, target)
                    models[m].compute_errors(activities, predictions, errors)

                    #Get accuracy over test set
                    accuracy = utilities.compute_num_correct(predictions[-1], y) / 10000

                    # Compute accuracy
                    models_accuracies[m].append(accuracy)


                #Find mean of each kind of data across models
                mean_accs = compute_means(models, models_accuracies)
                print(ep, mean_accs[-1])


            ############################################################################
            #Save each Epoch
            if record:
                if models[0].bp_train:
                    with open(f'data/BP_{d_name}.data','wb') as filehandle:
                        pickle.dump(models_accuracies, filehandle)
                else:
                    with open(f'data/PC_transpose{0+use_true_gradients}_KP{0+train_err_wts}_{num_iter}_{d_name}_{f_name}_online{0+train_online}_divErr{0+div_err}_posActs{0+pos_acts}_bias{0+(bias > 0)}.data', 'wb') as filehandle:
                        pickle.dump(models_accuracies, filehandle)
    if not record:
        return models_accuracies


#train_network(use_true_gradients=True, decay=0, num_models=1, num_iter=20, div_err=True, bot_infer_rate=.02)

#train_network(use_true_gradients=False, decay=0, num_models=1, num_iter=20, div_err=True, bot_infer_rate=.1)

#train_network(num_models=1,  bp_train=True)
