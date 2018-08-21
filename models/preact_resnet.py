'''Pre-activation ResNet in PyTorch.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Identity Mappings in Deep Residual Networks. arXiv:1603.05027
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import os
import torchvision
import torchvision.transforms as transforms

from torch.autograd import Variable


class PreActBlock(nn.Module):
    '''Pre-activation version of the BasicBlock.'''
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(PreActBlock, self).__init__()
	# nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)
	# 64; numero de canais da entrada, num de canais da saida
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
	# BatchNorm2d(num_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
	# Normalizacao tanto do input como do output... isso acelera o treinamento
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.bn2 = nn.BatchNorm2d(planes)
	
	# Ajuste para fazer funcionar um stride diferente de 1 (senao a convolucao nao funciona)
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        out = x   # --- 100x3x32x32
        out = self.bn1(out)
        relu1 = F.relu(out)   # --- Eh a funcao ReLU sendo aplicada.. torch.nnFunctional  100x64.32.32
        shortcut = self.shortcut(relu1) if hasattr(self, 'shortcut') else x
        out = self.conv1(relu1)
        out = self.bn2(out)
        relu2 = F.relu(out)
        out = self.conv2(relu2)
        out += shortcut
        return relu1, relu2, out


class PreActResNet(nn.Module):
    def __init__(self, block, num_classes=10,initial_planes=64,classes=10):
        super(PreActResNet, self).__init__()
        self.in_planes = initial_planes
        self.classes = classes

	# nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)
        self.conv1 = nn.Conv2d(3, self.in_planes, kernel_size=3, stride=1, padding=1, bias=False)

        self.layer1 = block(self.in_planes, self.in_planes, 1)
        self.layer2 = block(self.in_planes, self.in_planes, 1)

        self.layer3 = block(self.in_planes, self.in_planes*2, 2) # --- Muda a profundidade do bloco de saida e stride
        self.layer4 = block(self.in_planes*2, self.in_planes*2, 1)

        self.layer5 = block(self.in_planes*2, self.in_planes*4, 2)
        self.layer6 = block(self.in_planes*4, self.in_planes*4, 1)

        self.layer7 = block(self.in_planes*4, self.in_planes*8, 2)
        self.layer8 = block(self.in_planes*8, self.in_planes*8, 1)

        self.linear = nn.Linear(self.in_planes*8*block.expansion, self.classes)

        
    def forward(self, x, save=False, epoch=0,normalize=False):
        relus = list()
        if normalize:
            #relus.append(x)
            mean = torch.cuda.FloatTensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1)
            std = torch.cuda.FloatTensor([0.2023, 0.1994, 0.2010]).view(1, 3, 1, 1)
            x2 = x-torch.autograd.Variable(mean)
            x3 = x2/torch.autograd.Variable(std)
            out = self.conv1(x3)
        else:
            out = self.conv1(x)   # --- 100x3x32x32  --> 100x64x32x32
        out = self.layer1(out)
        relus.extend(out[:2])   # --- Vai concatenando os resultados da aplicacao da layer em out 
        out = self.layer2(out[2])   # --- A layer eh aplicada sempre em out[2] que eh de fato o out
        relus.extend(out[:2])        # --- Os elementos de out[0] e out[1] sao adcionados, relu1 e relu2

        out = self.layer3(out[2])
        relus.extend(out[:2])

        out = self.layer4(out[2])
        relus.extend(out[:2])        

        out = self.layer5(out[2])
        relus.extend(out[:2])        

        out = self.layer6(out[2])
        relus.extend(out[:2])

        out = self.layer7(out[2])
        relus.extend(out[:2])        

        out = self.layer8(out[2])
        relus.extend(out[:2])        
        
        final_out = F.relu(out[2])   # --- 
        relus.append(final_out)      # --- Coloca no final de relus o obnjeto final_out inteiro
	# avg_pool2d(input, kernel_size, stride=None, padding=0, ceil_mode=False, count_include_pad=True) 
        out = F.avg_pool2d(final_out, 4)   # --- Faz pooling usando a media dos valores
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return relus,out


def PreActResNet18(initial_planes=64,classes=10):   # Chamada na hora de criar a net
    return PreActResNet(PreActBlock,initial_planes=initial_planes,classes=classes)
