'''TEST ADVERSARIAL FOR CIFAR IMAGES'''
from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import random
import pandas as pd

import torchvision
import torchvision.transforms as transforms

import os
import argparse

from models import *
from utils import progress_bar
from torch.autograd import Variable
from gsp import force_smooth_network
import tqdm

torch.nn.Module.dump_patches = True
parser = argparse.ArgumentParser(description='PyTorch CIFAR Test Adversarial')
args = parser.parse_args()



use_cuda = torch.cuda.is_available()
torch.manual_seed(1)
if use_cuda:
    torch.cuda.manual_seed(1)

# Data
print('==> Preparing data..')
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

batch_size = 1
dataframeStarted = False

testset = torchvision.datasets.CIFAR10(root='/home/brain/pytorch-cifar/data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

folders = [
            ["Vanilla","results/False_False_False_0.0_0.0_1_0_False_0/"],
            #["Adversarial","results//"],
            #["Adversarial + Data Augmentation","results//"]
          ]

for name, folder in folders:

    print('==> Resuming from checkpoint.. {}'.format(folder))
    checkpoint = torch.load(folder + "ckpt.t7")
    net = checkpoint['net']

    if use_cuda:
        net.cuda()
        net = torch.nn.DataParallel(net, device_ids=[0])
        cudnn.benchmark = True

    criterion = nn.CrossEntropyLoss()


    net.eval()

    for wanted_snr in [33,40,50]:
        total = 0
        top1 = 0
        top1_under_attack = 0
        loss_func = nn.CrossEntropyLoss()
        snr = list()
        _diff_limit = list()
        for (ox, y) in tqdm.tqdm(testloader):
            total += ox.size()[0]
            ox = ox.cuda()
            x = Variable(ox, requires_grad=True)
            y = Variable(y.cuda())


            _, output = net(x,normalize=False)

            #attack
            net.zero_grad()
            loss = loss_func(output, y)
            loss.backward()

            diff_limit = (torch.norm(x).data/(10**(wanted_snr/20)))*((batch_size*3072)**-0.5)
#            diff_limit = 0.01
            _diff_limit.append(diff_limit)
            noise = diff_limit * torch.sign(x.grad.data)


            x2 = ox + noise
            x2 = Variable((x2.cuda()))
            x = Variable((ox.cuda()))

            division = (torch.norm(x)/torch.norm(noise)).data.cpu().numpy()

            _snr = 20*np.log10(division)
            snr.append(_snr)

            _, f2 = net(x,normalize=False)
            _, predicted = torch.max(f2, 1)
            top1 += float((predicted.data.cpu() == y.data.cpu()).sum())

            _, f2 = net(x2,normalize=False)
            _, predicted = torch.max(f2, 1)
            top1_under_attack += float((predicted.data.cpu() == y.data.cpu()).sum())
        print("Name", name)
        print("Top 1 Accuracy: {0:.3f}%".format(top1/total*100))
        print("Attacked: {0:.3f}%".format(top1_under_attack/total*100))
        print("SNR", np.array(snr).mean())
        print("Epsilon", np.array(_diff_limit).mean())
        
        result_dict = dict(folder=folder,name=name,
            accuracy=top1_under_attack/total*100,epsilon=wanted_snr)
        if not dataframeStarted:
            dataframe = pd.DataFrame(result_dict,index=[0])
            dataframeStarted = True
        else:
            dataframe = pd.concat([dataframe,pd.DataFrame(result_dict,index=[0])])
        #dataframe.to_csv(folder + "testAdversary_svhn.csv")
        dataframe.to_pickle(folder + "testAdversary_cifar.pkl")
