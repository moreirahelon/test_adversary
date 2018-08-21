'''Train CIFAR10 with PyTorch.'''
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

def create_adversary(inputs,targets,net,loss_func): #inputs : the images in dataset (X)
                                                    #targets : the result expected (Y)
                                                    #net : the network preact_resnet.py
                                                    #loss_func : the loss fuction (Cross-entropy + ...)

#   to apply the noise according to the signal of the gradient of x, moving away from zero.
#   In an image, the gradient points to the direction of change of intensity faster, that is where the noise will be added)
    _, output = net(inputs,normalize=False)         #runs the network receving the output
    #attack
    net.zero_grad()                                 #sets gradients of all model parameters to zero
    loss = loss_func(output, targets)               #loss calculation
    loss.backward()                                 #computes the gradient for every parameters which has requires_grad=True

# --- initialize the noise values ​​to be summed (Gaussian distribution values ​​to be applied in the direction of the gradient)
    noise_size = torch.ones(inputs.size()).cuda()   #a tensor of ones is create on GPU
    nn.init.normal(noise_size, 0, 0.0314)           #fills noise_size with a normal distributition(mean=0;standard deviation=0.0314)
    noise_size = torch.clamp(noise_size,-0.0628,0.0628) #use just the elements between -0.0628 and 0.0628 eliminating outliers
#    nn.init.normal(noise_size, 0, 0.007843137)
#    noise_size = torch.clamp(noise_size,-0.015686275,0.015686275)
    noise_size = torch.abs(noise_size)                #takes the absolute value of noise_size
    noise = noise_size * torch.sign(inputs.grad.data) #apply the sign (-1 or 1) of inputs.grad.data in noise_size

    x2 = inputs + Variable(noise)                  #apply the noise adversary in original inputs
    net.zero_grad()                                #reset to zero the gradients of all model parameters
    return x2



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


use_cuda = torch.cuda.is_available()
torch.manual_seed(args.seed)
if use_cuda:
    torch.cuda.manual_seed(args.seed)
best_acc = 0    #best test accuracy
start_epoch = 0 #start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')

if args.da:
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),   #cut out 32x32 size in the image at a random location
        transforms.RandomHorizontalFlip(),      #reverse the image horizontally with a probability of 0.5
        transforms.ToTensor(),
        #input[channel] = (input[channel] - mean[channel]) / std[channel]
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
else:
    if args.svhn:
        transform_train = transforms.Compose([
            transforms.ToTensor(),
    ])
    else:
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            #input[channel] = (input[channel] - mean[channel]) / std[channel]
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

if args.svhn:
    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])
else:
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

if args.svhn:
    trainset = torchvision.datasets.SVHN(root='/home/brain/pytorch-cifar/data', split="train", download=True, transform=transform_train)

    extraset = torchvision.datasets.SVHN(root='/home/brain/pytorch-cifar/data', split="extra", download=True, transform=transform_train)

    subset = torch.utils.data.Subset(extraset,np.arange(10000,extraset.data.shape[0]))
    trainset = torch.utils.data.ConcatDataset([trainset,subset])

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=100, shuffle=True, num_workers=2)

    testset = torchvision.datasets.SVHN(root='/home/brain/pytorch-cifar/data', split="test", download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)
else:
    if args.hundred:
        trainset = torchvision.datasets.CIFAR100(root='/home/brain/pytorch-cifar/data', train=True, download=True, transform=transform_train)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=100, shuffle=True, num_workers=2)

        testset = torchvision.datasets.CIFAR100(root='/home/brain/pytorch-cifar/data', train=False, download=True, transform=transform_test)
        testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

    else:
        trainset = torchvision.datasets.CIFAR10(root='/home/brain/pytorch-cifar/data', train=True, download=True, transform=transform_train)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=100, shuffle=True, num_workers=2)

        testset = torchvision.datasets.CIFAR10(root='/home/brain/pytorch-cifar/data', train=False, download=True, transform=transform_test)
        testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

if args.beta > 0:
    net = PreActResNet18Parseval()
else:
    net = PreActResNet18()
if use_cuda:
    net.cuda()
    net = torch.nn.DataParallel(net, device_ids=[0])
    cudnn.benchmark = True

criterion = nn.CrossEntropyLoss()
path = "results/{}_{}_{}_{}_{}_{}_{}_{}_{}/".format(args.adversary,args.hundred,args.da,args.beta,args.gamma,args.m,args.k,args.svhn,args.seed)
try:
    os.makedirs(path)
except:
    pass

params = net.parameters()
parseval_parameters = list()
for param in params:
    if len(param.size()) > 1:
        parseval_parameters.append(param)

def do_parseval(parseval_parameters):
    for W in parseval_parameters:
        Ws = W.view(W.size(0),-1)
        W_partial = Ws.data.clone()
        W_partial = (1+args.beta)*W_partial - args.beta*(torch.mm(torch.mm(W_partial,torch.t(W_partial)),W_partial))
        new = W_partial
        new = new.view(W.size())
        W.data.copy_(new)

dataframeStarted = False
dataframeStarted2 = False
dataframe,dataframe2 = None,None
# Training
def train(epoch, optimizer):
    global dataframeStarted,dataframe
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss2 = 0
    train_loss1 = 0
    train_loss = 0
    correct = 0.
    total = 0.

    for batch_idx, (inputs, targets) in enumerate(trainloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        optimizer.zero_grad()
        if args.adversary:
            targets = Variable(targets)
            inputs1, targets1 = Variable(inputs[:50],requires_grad=True), targets[:50]
            inputs1 = create_adversary(inputs1,targets1,net,criterion)
            relus1, outputs1 = net(inputs1)

            inputs2, targets2 = Variable(inputs[50:]), targets[50:]
            relus2, outputs2 = net(inputs2)
            relus = list()
            for a,b in zip(relus1,relus2):
                relus.append(torch.cat([a,b],0))
            outputs = torch.cat([outputs1, outputs2], 0)

            loss1 = 50*criterion(outputs1, targets1) + 50*0.3 * criterion(outputs2,targets2)
            loss1 = loss1/(50+(50*0.3))
        else:
            inputs, targets = Variable(inputs), Variable(targets)
            relus, outputs = net(inputs)
            loss1 = criterion(outputs, targets)
        if args.gamma > 0:
            if args.k > 0:
                loss2 = force_smooth_network(relus,targets,m=args.m,k=args.k)
            else:
                loss2 = force_smooth_network(relus,targets,m=args.m)
            value = 1/args.gamma
            loss = loss1 + loss2/(value**args.m)
            train_loss2 += loss2.item()
        else:
            loss2 = 0
            train_loss2 += loss2
            loss = loss1
        loss.backward()
        optimizer.step()
        if args.beta > 0:
            do_parseval(parseval_parameters)

        train_loss += loss.item()
        train_loss1 += loss1.item()
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (train_loss/(batch_idx+1),
            100.*correct/total, correct, total))

#        progress_bar(batch_idx, len(trainloader), 'Log Loss: %.3f | Smooth Loss: %.3f | Loss: %.3f | Acc: %.3f%% (%d/%d)'
#            % (train_loss1/(batch_idx+1),train_loss2/(batch_idx+1),train_loss/(batch_idx+1),
#            100.*correct/total, correct, total))
    f = open(path + 'score_training.txt','a')
    f.write(str(1.*correct/total))
    f.write('\n')
    f.close()

    result_train_dict = dict(log_loss=train_loss1/(batch_idx+1),smooth_loss=train_loss2/(batch_idx+1),loss=train_loss/(batch_idx+1),accuracy=100.*correct/total,dataset="train")

    if not dataframeStarted:
            dataframe = pd.DataFrame(result_train_dict,index=[epoch])
            dataframeStarted = True
    else:
        dataframe = pd.concat([dataframe,pd.DataFrame(result_train_dict,index=[epoch])])
    dataframe.to_pickle(path + "result_dict.pkl")   #while the code is running we can already see how it's working

def test(epoch):
    print('\nTest... epoch %d' % epoch)
    global best_acc, dataframe
    net.eval()
    test_loss = 0
    test_loss1 = 0
    test_loss2 = 0
    correct = 0.
    total = 0.

    for batch_idx, (inputs, targets) in enumerate(testloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = Variable(inputs), Variable(targets)

        relus, outputs = net(inputs)
        loss1 = criterion(outputs, targets)
        if args.gamma > 0:
            if args.k > 0:
                loss2 = force_smooth_network(relus,targets,m=args.m,k=args.k)
            else:
                loss2 = force_smooth_network(relus,targets,m=args.m)
            value = 1/args.gamma
            loss = loss1 + loss2/(value**args.m)
            test_loss2 += loss2.item()
        else:
            loss2 = 0
            test_loss2 += loss2
            loss = loss1
        test_loss += loss.item()
        test_loss1 += loss1.item()
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum().item()

        progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

#        progress_bar(batch_idx, len(testloader), 'Log Loss: %.3f | Smooth Loss: %.3f | Loss: %.3f | Acc: %.3f%% (%d/%d)'
#            % (test_loss1/(batch_idx+1),test_loss2/(batch_idx+1),test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    acc = 100.*correct/total
    f = open(path + 'score.txt','a')
    f.write(str(1.*correct/total))
    f.write('\n')
    f.close()

    result_test_dict = dict(log_loss=test_loss1/(batch_idx+1),smooth_loss=test_loss2/(batch_idx+1),
    loss=test_loss/(batch_idx+1),accuracy=100.*correct/total,dataset="test")
    dataframe = pd.concat([dataframe,pd.DataFrame(result_test_dict,index=[epoch])])
    dataframe.to_pickle(path + "result_dict.pkl")   #During the code is running we can already see how it's working

def save(epoch):
    net.forward(examples, True, epoch)

def save_model():
    state = {
        'net': net.module if use_cuda else net,
    }
    torch.save(state, path+'/ckpt.t7')

f = open(path + 'score.txt','w')
f.write("0.1\n")
f.close()
f = open(path + 'score_training.txt','w')
f.write("0.1\n")
f.close()

if args.da:
    epoch_start = [0,150,250]
    epoch_end =  [150,250,350]
    for period in range(3):
        if period == 0:
            optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
        elif period == 1:
            optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
        else:
            optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)

        for epoch in range(epoch_start[period], epoch_end[period]):
            train(epoch, optimizer)
            test(epoch)
else:
    if args.svhn:
        epoch_start = [0,5] #the graph of loss becomes horizontal really fast with lr=0.1
        epoch_end =  [5,55]
        for period in range(2):
            if period == 0:
                optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
            else:
                optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)

            for epoch in range(epoch_start[period], epoch_end[period]):
                train(epoch, optimizer)
                test(epoch)
    else:
        for period in range(2):
            if period == 0:
                optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
            else:
                optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)

            for epoch in range(50*period, 50*(period+1)):
                train(epoch, optimizer)
                test(epoch)

save_model()
#save(epoch)

