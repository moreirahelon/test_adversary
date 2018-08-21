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

from torch.autograd import Variable

class Parseval_Conv2d(nn.Conv2d):

    def forward(self, input):
        new_weight = self.weight/np.sqrt(2*self.kernel_size[0]*self.kernel_size[1]+1)
        return F.conv2d(input, new_weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)


class PreActBlock(nn.Module):
    '''Pre-activation version of the BasicBlock.'''
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(PreActBlock, self).__init__()
        self.conv1 = Parseval_Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.conv2 = Parseval_Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.bn2 = nn.BatchNorm2d(planes)

        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                Parseval_Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        out = x
        out = self.bn1(out)
        relu1 = F.relu(out)
        shortcut = self.shortcut(relu1) if hasattr(self, 'shortcut') else x
        out = self.conv1(relu1)
        out = self.bn2(out)
        relu2 = F.relu(out)
        out = self.conv2(relu2)
        out = 0.5*out + 0.5*shortcut
        return relu1, relu2, out


class PreActResNetParseval(nn.Module):
    def __init__(self, block, num_classes=10):
        super(PreActResNetParseval, self).__init__()
        self.in_planes = 64


        self.conv1 = Parseval_Conv2d(3, self.in_planes, kernel_size=3, stride=1, padding=1, bias=False)

        self.layer1 = block(self.in_planes, self.in_planes, 1)
        self.layer2 = block(self.in_planes, self.in_planes, 1)

        self.layer3 = block(self.in_planes, self.in_planes*2, 2)
        self.layer4 = block(self.in_planes*2, self.in_planes*2, 1)

        self.layer5 = block(self.in_planes*2, self.in_planes*4, 2)
        self.layer6 = block(self.in_planes*4, self.in_planes*4, 1)

        self.layer7 = block(self.in_planes*4, self.in_planes*8, 2)
        self.layer8 = block(self.in_planes*8, self.in_planes*8, 1)

        self.linear = nn.Linear(self.in_planes*8*block.expansion, num_classes)

        
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
            out = self.conv1(x)
        out = self.layer1(out)
        relus.extend(out[:2])        
        out = self.layer2(out[2])
        relus.extend(out[:2])        

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
        
        final_out = F.relu(out[2])
        relus.append(final_out)        
        out = F.avg_pool2d(final_out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return relus,out


def PreActResNet18Parseval():
    return PreActResNetParseval(PreActBlock)
