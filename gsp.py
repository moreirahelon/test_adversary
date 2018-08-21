import torch
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F

def force_smooth_network(relus,targets,m=1,classes=10,k=None):
    targets = targets.clone()
    laplacians = list()
    for i, relu in enumerate(relus):
        result = laplacian(relu,targets,m=m,classes=classes,k=k)
        laplacians.append(result)
    loss = 0
    for i in range(len(laplacians)):
        if i == 0:
            continue
        else:
            result = torch.abs(laplacians[i]-laplacians[i-1])
            loss += result
    return loss/((len(laplacians)-1))

def laplacian(x_values,target,distance="cosine",m=1,classes=10,k=None,extract=False,reg_l2=False):
    x_values = x_values.clone()
    target = target.clone()

    n_examples = x_values.size(0)
    x_values = x_values.view(n_examples,-1)


    y_true = torch.cuda.FloatTensor(n_examples,classes)
    y_true.zero_()
    y_true.scatter_(1, target.data.view(-1,1), 1)
    y_true = Variable(y_true)
    
    transposed_y_true = torch.t(y_true)
    if k is None:
        neighbours = n_examples
    else:
        neighbours = k        

    if distance == "cosine":
        normalized =  F.normalize(x_values, p=2, dim=1)
        W_tf = torch.mm(normalized,torch.t(normalized))
    
    if neighbours != n_examples:
        y, ind = torch.sort(W_tf, 1)
        A = torch.zeros(*y.size()).cuda()
        k_biggest = ind[:,-neighbours:].data
        for index1,value in enumerate(k_biggest):
            A_line = A[index1]
            A_line[value] = 1
        A_final = Variable(torch.min(torch.ones(*y.size()).cuda(),A+torch.t(A)))
        new_W_tf = W_tf*A_final
    else:
        new_W_tf = W_tf
    
    d_tf = torch.sum(new_W_tf,1)
    d_tf = torch.diag(d_tf)
    laplacian_tf = (d_tf - new_W_tf)
    laplacian_after_m = laplacian_tf
    for _ in range(1,m):
        laplacian_after_m = torch.mm(laplacian_after_m,laplacian_tf)
    if reg_l2 and m > 1:
        clone = torch.abs(laplacian_after_m.clone())
        mask = torch.diag(torch.ones_like(clone[0]))
        clone *= (1-mask)
        max_val = torch.max(clone.view(-1))
        laplacian_after_m /= max_val 
    if extract:
        return laplacian_after_m
    else:
        final_laplacian_tf = torch.mm(transposed_y_true, laplacian_after_m)
        final_laplacian_tf = torch.mm(final_laplacian_tf,y_true)
        final_laplacian_tf = torch.trace(final_laplacian_tf)
        return final_laplacian_tf
