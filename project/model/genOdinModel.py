"""Generalized Odin in PyTorch.
Self developed implementation
Reference:
[1] Yen-Chang Hsu, Yilin Shen, Hongxia Jin, Zsolt Kira

arxiv: https://arxiv.org/abs/2002.11297
"""


import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch


class euc_dist_layer(nn.Module):
    def __init__(self, out_classes, dimensions):
        super().__init__()
        # weigths = torch.Tensor((1, out_classes, dimensions), dtype=torch.float64)
        weigths = torch.rand(
            (1, out_classes, dimensions), dtype=torch.float32, requires_grad=True
        )  # 1 10 64

        self.weigths = nn.Parameter(weigths)  # size num_classes, weight  1 10 64

    def forward(self, x):  # 128 64
        # https://pytorch.org/docs/stable/generated/torch.lisnalg.norm.html#torch.linalg.norm
        x = x.unsqueeze(
            dim=0
        )  # (batch, extra dim, data) dim=0 1 128 64 / dim=-2 128 1 64
        x -= self.weigths
        return torch.linalg.norm(x, dim=-1)


class cosine_layer(nn.Module):
    def __init__(self, out_classes, dimensions):
        super().__init__()
        # weigths = torch.Tensor((out_classes, dimensions), dtype=torch.float64)
        weigths = torch.rand(
            (out_classes, dimensions), dtype=torch.float32, requires_grad=True
        )
        self.weigths = nn.Parameter(weigths)  # 10 64

    def forward(self, x, eps=1e-08):  # 128 64
        # https://pytorch.org/docs/stable/generated/torch.nn.CosineSimilarity.html
        # x =>  batch , D
        eps = torch.tensor(eps, dtype=torch.float32)

        x_norm = torch.linalg.norm(x, dim=0)  # -1:128 -- now 0:64
        w_norm = torch.linalg.norm(
            self.weigths, dim=0
        )  # .unsqueeze(dim=0) # -1:0:1 10 -- 0:64

        nominator = torch.matmul(x, self.weigths.T)  # 128 10
        denominator = torch.max((torch.matmul(x_norm, w_norm)), eps)  # empty (1?)

        return torch.div(nominator, denominator)


class genOdinModel(nn.Module):
    """Net [General Odin implementation]"""

    def __init__(
        self,
        activation=F.relu,
        similarity="C",
        out_classes=10,
        include_bn=False,
        channel_input=3,
    ):
        super(genOdinModel, self).__init__()

        self.conv1 = nn.Conv2d(channel_input, 4, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(4, 12, 5)
        self.fc1 = nn.Linear(12 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 64)
        self.fc3 = nn.Linear(64, 10)
        self.dropout1 = nn.Dropout(p=0.4)
        self.dropout2 = nn.Dropout(p=0.4)
        self.bn1 = nn.BatchNorm1d(self.fc1.in_features)
        self.bn2 = nn.BatchNorm1d(self.fc1.out_features)
        self.out_classes = out_classes
        self.flatten = nn.Flatten()

        self.activation = activation
        self.similarity = similarity
        self.include_bn = include_bn

        self.g_activation = nn.Sigmoid()
        self.g_func = nn.Linear(self.fc2.out_features, 1)
        self.g_norm = nn.BatchNorm1d(self.g_func.out_features)

        if self.similarity == "I":
            self.dropout_3 = nn.Dropout(p=0.6)
            self.h_func = nn.Linear(64, 10)

        elif self.similarity == "E":
            self.dropout_3 = nn.Dropout(0)
            self.h_func = euc_dist_layer(out_classes, self.fc2.out_features)

        elif self.similarity == "C":
            self.dropout_3 = nn.Dropout(p=0)
            self.h_func = cosine_layer(out_classes, self.fc2.out_features)
        else:
            assert False, "Incorrect similarity Measure"

    def forward(self, input_data, get_test_model=True):  # 128 , 3 ,32, 32
        x = self.pool(self.activation(self.conv1(input_data)))  # 128 , 4 , 14, 14
        x = self.pool(self.activation(self.conv2(x)))  # 128, 12, 5,5
        # x = x.view(-1, 12 * 5 * 5)  # 200, 192
        x = self.flatten(x)  # 128 300
        if self.include_bn:
            x = self.bn1(x)
        x = self.activation(self.fc1(x))  # 128 300
        if self.include_bn:
            x = self.bn2(x)
        if self.similarity == "I":
            x = self.activation(self.fc2(x))
        else:
            x = self.fc2(x)  # 128 64

        g = self.g_activation(self.g_norm(self.g_func(x)))  # scalar
        h = self.h_func(x)  # 128 10
        out = torch.div(g, h)
        if get_test_model:
            return out
        else:
            return out, g, h  # 128,10 ; 1 : 128,10
