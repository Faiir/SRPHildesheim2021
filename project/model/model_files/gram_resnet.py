import sys
from tqdm import tqdm

import random
import math

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.init as init
from torch.autograd import Variable, grad
from torchvision import datasets, transforms
from torch.nn.parameter import Parameter

import warnings
warnings.filterwarnings('ignore')


def get_gram_resnet():
    def conv3x3(in_planes, out_planes, stride=1):
        return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


    class LambdaLayer(nn.Module):
        def __init__(self, lambd):
            super(LambdaLayer, self).__init__()
            self.lambd = lambd

        def forward(self, x):
            return self.lambd(x)


    class BasicBlock(nn.Module):
        expansion = 1

        def __init__(self, in_planes, planes, stride=1, option='A'):
            super(BasicBlock, self).__init__()
            self.conv1 = conv3x3(in_planes, planes, stride)
            self.bn1 = nn.BatchNorm2d(planes)
            self.conv2 = conv3x3(planes, planes)
            self.bn2 = nn.BatchNorm2d(planes)

            self.shortcut = nn.Sequential()
            if stride != 1 or in_planes != planes:
                if option == "A":
                    """
                    For CIFAR10 ResNet paper uses option A.
                    """
                    self.shortcut = LambdaLayer(
                        lambda x: F.pad(
                            x[:, :, ::2, ::2],
                            (0, 0, 0, 0, planes // 4, planes // 4),
                            "constant",
                            0,
                        )
                    )
                elif option == "B":
                    self.shortcut = nn.Sequential(
                        nn.Conv2d(
                            in_planes,
                            self.expansion * planes,
                            kernel_size=1,
                            stride=stride,
                            bias=False,
                        ),
                        nn.BatchNorm2d(self.expansion * planes),
                    )
        
        def forward(self, x):
            t = self.conv1(x)
            out = F.relu(self.bn1(t))
            torch_model.record(t)
            torch_model.record(out)
            t = self.conv2(out)
            out = self.bn2(t)
            torch_model.record(t)
            torch_model.record(out)
            t = self.shortcut(x)
            out += t
            torch_model.record(t)
            out = F.relu(out)
            torch_model.record(out)
            
            return out

    class ResNet(nn.Module):
        def __init__(self, block, num_blocks, num_classes=10):
            super(ResNet, self).__init__()
            self.in_planes = 16

            self.conv1 = conv3x3(3, 16, 1)
            self.bn1 = nn.BatchNorm2d(16)
            self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
            self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
            self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
            self.linear = nn.Linear(64, num_classes)
            
            self.collecting = False
            self.has_weighing_factor = False
            self.softmax = nn.Softmax(dim=-1)
        
        def _make_layer(self, block, planes, num_blocks, stride):
            strides = [stride] + [1] * (num_blocks - 1)
            layers = []
            for stride in strides:
                layers.append(block(self.in_planes, planes, stride))
                self.in_planes = planes * block.expansion

            return nn.Sequential(*layers)
        
        def forward(self, x, apply_softmax=False):
            out = F.relu(self.bn1(self.conv1(x)))
            out = self.layer1(out)
            out = self.layer2(out)
            out = self.layer3(out)
            out = F.avg_pool2d(out, out.size()[3])
            out = out.view(out.size(0), -1)
            y = self.linear(out)

            if apply_softmax:
                y = self.softmax(y)
            return y
        
        def record(self, t):
            if self.collecting:
                self.gram_feats.append(t)
        
        def gram_feature_list(self,x):
            if torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"
            self.collecting = True
            self.gram_feats = []
            self.forward(x.to(device))
            self.collecting = False
            temp = self.gram_feats
            self.gram_feats = []
            return temp
        
        def load(self, path="resnet_cifar10.pth"):
            tm = torch.load(path,map_location="cpu")        
            self.load_state_dict(tm)
        
        def get_min_max(self, data, power):
            mins = []
            maxs = []

            if torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"
            
            #data = 

            for i in range(0,len(data),128):
                batch = torch.stack(data[i:i+128]).to(device)

                feat_list = self.gram_feature_list(batch)
                for L,feat_L in enumerate(feat_list):
                    if L==len(mins):
                        mins.append([None]*len(power))
                        maxs.append([None]*len(power))
                    
                    for p,P in enumerate(power):
                        g_p = G_p(feat_L,P)
                        
                        current_min = g_p.min(dim=0,keepdim=True)[0]
                        current_max = g_p.max(dim=0,keepdim=True)[0]
                        
                        if mins[L][p] is None:
                            mins[L][p] = current_min
                            maxs[L][p] = current_max
                        else:
                            mins[L][p] = torch.min(current_min,mins[L][p])
                            maxs[L][p] = torch.max(current_max,maxs[L][p])
            
            return mins,maxs
        
        def get_deviations(self,data,power,mins,maxs):
            deviations = []
            if torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"

            for i in range(0,len(data),128):
                batch = data[i:i+128][0].to(device)

                feat_list = self.gram_feature_list(batch)
                batch_deviations = []
                for L,feat_L in enumerate(feat_list):
                    dev = 0
                    for p,P in enumerate(power):
                        g_p = G_p(feat_L,P)
                        
                        dev +=  (F.relu(mins[L][p]-g_p)/torch.abs(mins[L][p]+10**-6)).sum(dim=1,keepdim=True)
                        dev +=  (F.relu(g_p-maxs[L][p])/torch.abs(maxs[L][p]+10**-6)).sum(dim=1,keepdim=True)
                    batch_deviations.append(dev.cpu().detach().numpy())
                batch_deviations = np.concatenate(batch_deviations,axis=1)
                deviations.append(batch_deviations)
            deviations = np.concatenate(deviations,axis=0)
            
            return deviations


    torch_model = ResNet(BasicBlock, [3, 3, 3], num_classes=10)
    return torch_model

def G_p(ob, p):
    temp = ob.detach()
    
    temp = temp**p
    temp = temp.reshape(temp.shape[0],temp.shape[1],-1)
    temp = ((torch.matmul(temp,temp.transpose(dim0=2,dim1=1)))).sum(dim=2) 
    temp = (temp.sign()*torch.abs(temp)**(1/p)).reshape(temp.shape[0],-1)
    
    return temp


def cpu(ob):
    for i in range(len(ob)):
        for j in range(len(ob[i])):
            ob[i][j] = ob[i][j].cpu()
    return ob

def cuda(ob):
    for i in range(len(ob)):
        for j in range(len(ob[i])):
            ob[i][j] = ob[i][j].cuda()
    return ob


def detect(all_test_deviations,all_ood_deviations, verbose=True, normalize=True):
    average_results = {}
    for i in range(1,11):
        random.seed(i)
        
        validation_indices = random.sample(range(len(all_test_deviations)),int(0.1*len(all_test_deviations)))
        test_indices = sorted(list(set(range(len(all_test_deviations)))-set(validation_indices)))

        validation = all_test_deviations[validation_indices]
        test_deviations = all_test_deviations[test_indices]

        t95 = validation.mean(axis=0)+10**-7
        if not normalize:
            t95 = np.ones_like(t95)
        test_deviations = (test_deviations/t95[np.newaxis,:]).sum(axis=1)
        ood_deviations = (all_ood_deviations/t95[np.newaxis,:]).sum(axis=1)
        
        results = callog.compute_metric(-test_deviations,-ood_deviations)
        for m in results:
            average_results[m] = average_results.get(m,0)+results[m]
    
    for m in average_results:
        average_results[m] /= i
    if verbose:
        callog.print_results(average_results)
    return average_results



class Detector:
    def __init__(self, num_classes=10):
        self.all_test_deviations = None
        self.mins = {}
        self.maxs = {}
        
        self.classes = range(num_classes)
    
    def compute_minmaxs(self,torch_model,train_loader,POWERS=[10]):
        train_preds = []
        train_confs = []
        train_logits = []
        data_train = train_loader.dataset
        #print(data_train[0:15])
        if torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"

        for idx in range(0,len(data_train),128):
            batch = torch.squeeze(data_train[idx:idx+128][0]).to(device)
            logits = torch_model(batch)
            #confs = F.softmax(logits,dim=1).cpu().detach().numpy()
            preds = np.argmax(logits.cpu().detach().numpy(),axis=1)

            train_preds.extend(preds)

        for PRED in tqdm(self.classes):
            train_indices = np.where(np.array(train_preds)==PRED)[0]
            train_PRED = ([data_train[i][0] for i in train_indices])
            mins,maxs = torch_model.get_min_max(train_PRED,power=POWERS)
            self.mins[PRED] = cpu(mins)
            self.maxs[PRED] = cpu(maxs)
            torch.cuda.empty_cache()

    
    def compute_deviations(self,torch_model,data_loader,POWERS=[10]):
        assert len(self.mins)>0, 'Run compute_min_max first to generate mins and maxs'
        test_preds = []
        test_confs = []
        test_logits = []
        if torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"

        data = data_loader.dataset
        for idx in range(0,len(data),128):
            batch =  torch.squeeze(data[idx:idx+128][0]).to(device)
            
            logits = torch_model(batch)
            confs = F.softmax(logits,dim=1).cpu().detach().numpy()
            preds = np.argmax(confs,axis=1)
            logits = (logits.cpu().detach().numpy())

            test_confs.extend(np.max(confs,axis=1))    
            test_preds.extend(preds)



        all_deviations = None
        for PRED in tqdm(self.classes):
            test_indices = np.where(np.array(test_preds)==PRED)[0]
            test_PRED = [data[i][0] for i in test_indices]
            test_confs_PRED = np.array([test_confs[i] for i in test_indices])
            
            
            mins = cuda(self.mins[PRED])
            maxs = cuda(self.maxs[PRED])
            test_deviations = torch_model.get_deviations(test_PRED,power=POWERS,mins=mins,maxs=maxs)/test_confs_PRED[:,np.newaxis]
            cpu(mins)
            cpu(maxs)
            if all_test_deviations is None:
                all_test_deviations = test_deviations
            else:
                all_test_deviations = np.concatenate([all_test_deviations,test_deviations],axis=0)
            torch.cuda.empty_cache()

        return all_test_deviations
