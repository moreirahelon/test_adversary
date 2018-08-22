import pandas as pd
import matplotlib.pyplot as plt

import numpy as np
import os
import argparse

import plotly
import plotly.plotly as py
import plotly.graph_objs as go
import plotly.figure_factory as FF

import pickle as pkl
import os

parser = argparse.ArgumentParser(description='PyTorch CIFAR Training')
parser.add_argument('-m', default=1, type=int, help='laplacian power')
parser.add_argument('-k', default=0, type=int, help='number of neighbors')
parser.add_argument('--beta', default=0., type=float, help='parseval beta parameter')
parser.add_argument('--gamma', default=0., type=float, help='laplacian weight parameter')
parser.add_argument('--da', action='store_true', help='data augmentation')
parser.add_argument('--adversary', action='store_true', help='adversarial noise')
parser.add_argument('--hundred', action='store_true', help='use cifar100 instead of cifar10')
parser.add_argument('--svhn', action='store_true', help='use svhn instead of cifar10')
parser.add_argument('--seed', default=0, type=int, help='seed')
args = parser.parse_args()

    #Don't forget to use the correct folder for each case !
path = "results/{}_{}_{}_{}_{}_{}_{}_{}_{}/".format(args.adversary,args.hundred,args.da,args.beta,args.gamma,args.m,args.k,args.svhn,args.seed)

try:
    data = pd.read_pickle(path + "result_dict.pkl")
except:
    raise Exception("The path doesn't exist!\n")
data_train = data.query("dataset=='train'")
data_test = data.query("dataset=='test'")

    #plot only loss   -------------------------------
fig, (ax1,ax2) = plt.subplots(1,2)
fig.set_size_inches(20, 8, forward=True)
data_train.plot(y="loss",ax=ax1, title="Train", grid = True)
data_test.plot(y="loss",ax=ax2, title="Test", grid = True)
plt.savefig(os.path.join(path, 'loss'), dpi = 500,format = 'png', frameon = False)

    #plot only logloss   -------------------------------
fig, (ax1,ax2) = plt.subplots(1,2)
fig.set_size_inches(20, 8, forward=True)
data_train.plot(y="log_loss",ax=ax1, title="Train", grid = True)
data_test.plot(y="log_loss",ax=ax2, title="Test", grid = True)
plt.savefig(os.path.join(path, 'log_loss'),dpi = 500, format = 'png', frameon = False)

    #plot only smooth loss   -------------------------------
fig, (ax1,ax2) = plt.subplots(1,2)
fig.set_size_inches(20, 8, forward=True)
data_train.plot(y="smooth_loss",ax=ax1, title="Train", grid = True)
data_test.plot(y="smooth_loss",ax=ax2, title="Test", grid = True)
plt.savefig(os.path.join(path, 'smooth_loss'), dpi = 500, format = 'png', frameon = False)

    #plot only accuracy   -------------------------------
fig, (ax1,ax2) = plt.subplots(1,2)
fig.set_size_inches(20, 8, forward=True)
data_train.plot(y="accuracy",ax=ax1, title="Train", grid = True)
data_test.plot(y="accuracy",ax=ax2, title="Test", grid = True)
plt.savefig(os.path.join(path, 'accuracy'), dpi = 500, format = 'png', frameon = False)

#plt.show()

