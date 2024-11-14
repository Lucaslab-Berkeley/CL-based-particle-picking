import numpy as np
import matplotlib.pyplot as plt

def plot_classifier_acc(losses_train=None, losses_val=None, type_='Accuracy', return_fig= False):
    '''
        Function to plot metrics
            Args:
                losses_val   : Validation losses of each epoch | list
                losses_train : Train losses of each epoch | list
                type         : Metric type to plot | loss/TM/LDDT
    '''
    plt.figure()
    plt.plot(losses_val, label= f'val {type_}')
    if losses_train != None:
        plt.plot(losses_train, label= f'train {type_}')
    # plt.axhline(y = train_tm_lddt, color = 'r', linestyle = '-')
    plt.legend()
    plt.title(f'{type_}')
    
    if not return_fig:plt.show()
    
def plot_train_losses(losses_train, losses_val=None, type_='loss', return_fig= False):
    '''
        Function to plot metrics
            Args:
                losses_train : Train losses of each epoch | list
                type         : Metric type to plot | loss/TM/LDDT
    '''
    plt.figure()
    if losses_val != None:
        plt.plot(losses_val, label= f'val {type_}')
    plt.plot(losses_train, label= f'train {type_}')
    # plt.axhline(y = train_tm_lddt, color = 'r', linestyle = '-')
    plt.legend()
    plt.title(f'{type_}')
    
    if not return_fig:plt.show()