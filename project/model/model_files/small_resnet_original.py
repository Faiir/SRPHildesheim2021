"""
Properly implemented ResNet-s for CIFAR10 as described in paper [1].

The implementation and structure of this file is hugely influenced by [2]
which is implemented for ImageNet and doesn't have option A for identity.
Moreover, most of the implementations on the web is copy-paste from
torchvision's resnet and has wrong number of params.

Proper ResNet-s for CIFAR10 (for fair comparision and etc.) has following
number of layers and parameters:

name      | layers | params
ResNet20  |    20  | 0.27M
ResNet32  |    32  | 0.46M
ResNet44  |    44  | 0.66M
ResNet56  |    56  | 0.85M
ResNet110 |   110  |  1.7M
ResNet1202|  1202  | 19.4m

which this implementation indeed has.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
[2] https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py

If you use this implementation in you work, please don't forget to mention the
author, Yerlan Idelbayev.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from torch.autograd import Variable

__all__ = [
    "ResNet",
    "resnet20",
    "resnet32",
    "resnet44",
    "resnet56",
    "resnet110",
    "resnet1202",
]


def _weights_init(m):
    classname = m.__class__.__name__
    # print(classname)
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)


class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class euc_dist_layer(nn.Module):
    def __init__(self, in_dimensions, out_dimensions):
        super().__init__()
        if torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"

        weights = torch.empty((1, in_dimensions, out_dimensions), dtype=torch.float32)

        self.weights = nn.Parameter(weights)
        init.kaiming_normal_(self.weights)

    def forward(self, x):
        x = x.unsqueeze(dim=-1)
        diff = x - self.weights
        diff_squared = torch.square(diff)
        diff_summed = torch.sum(diff_squared, dim=1)
        out = torch.sqrt(diff_summed)
        return out


class euc_dist_layer_corrected(nn.Module):
    def __init__(self, in_dimensions, out_dimensions):
        super().__init__()
        if torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"

        weights = torch.empty((1, in_dimensions, out_dimensions), dtype=torch.float32)

        self.weights = nn.Parameter(weights)
        init.kaiming_normal_(self.weights)

    def forward(self, x):
        x = x.unsqueeze(dim=-1)
        diff = x - self.weights
        diff_squared = torch.square(diff)
        diff_summed = torch.sum(diff_squared, dim=1)
        out = -torch.sqrt(diff_summed)
        return out


class cosine_layer(nn.Module):
    def __init__(self, in_dimensions, out_dimensions):
        super().__init__()
        if torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"

        weights = torch.empty((1, in_dimensions, out_dimensions), dtype=torch.float32)

        self.weights = nn.Parameter(weights)
        init.kaiming_normal_(self.weights)

    def forward(self, x):
        x = x.unsqueeze(-1)
        cos = nn.CosineSimilarity(dim=1)
        return cos(self.weights, x)


class cosine_layer_holy(nn.Module):
    def __init__(self, in_dimensions, out_dimensions):
        super().__init__()
        if torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"

        weights = torch.empty((1, in_dimensions, out_dimensions), dtype=torch.float32)

        self.weights = nn.Parameter(weights)
        init.kaiming_normal_(self.weights)

    def forward(self, x):
        x = x.unsqueeze(-1)
        cos = nn.CosineSimilarity(dim=1)
        return -cos(self.weights, x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option="A"):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
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
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, similarity=None):
        super(ResNet, self).__init__()
        self.in_planes = 16
        self.similarity = similarity

        if self.similarity is None:
            print("INFO ----- ResNet has been initialized without a similarity measure")
        else:
            print(
                f"INFO ----- ResNet has been initialized with a similarity measure : {self.similarity}"
            )

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.softmax = nn.Softmax()

        if self.similarity is None:
            self.linear = nn.Linear(64, num_classes)
        else:
            self.g_activation = nn.Sigmoid()
            self.g_func = nn.Linear(64, 1)
            self.g_norm = nn.BatchNorm1d(self.g_func.out_features)

            if "I" in self.similarity:
                self.h_func = nn.Linear(64, num_classes)
            elif "E" in self.similarity:
                self.h_func = euc_dist_layer(64, num_classes)
            elif "C" in self.similarity:
                self.h_func = cosine_layer(64, num_classes)

            if "E_U" in self.similarity:
                self.h_func = euc_dist_layer_corrected(64, num_classes)
            elif "C_H" in self.similarity:
                self.h_func = cosine_layer_holy(64, num_classes)

            if "R" in self.similarity:
                self.scaling_factor = nn.Parameter(torch.Tensor(1, 1))
                init.kaiming_normal_(self.scaling_factor)

        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x, get_test_model=False):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        if self.similarity is None:
            out = self.linear(out)
        else:
            h = self.h_func(out)
            g = self.g_func(out)
            g = self.g_norm(g)
            g = self.g_activation(g)

            if "R" in self.similarity:
                scale = torch.add(1, torch.mul(self.scaling_factor.exp(), 1 - g))
                out = torch.mul(h, scale)
            else:
                out = torch.div(h, g)

        if get_test_model:
            return out, g, h
        else:
            return out


def resnet20(num_classes, similarity):
    return ResNet(BasicBlock, [3, 3, 3], num_classes, similarity)


def resnet32(num_classes, similarity):
    return ResNet(BasicBlock, [5, 5, 5], num_classes, similarity)


def resnet44(num_classes, similarity):
    return ResNet(BasicBlock, [7, 7, 7].num_classes, similarity)


def resnet56(num_classes, similarity):
    return ResNet(BasicBlock, [9, 9, 9], num_classes, similarity)


def resnet110(num_classes, similarity):
    return ResNet(BasicBlock, [18, 18, 18], num_classes, similarity)


def resnet1202(num_classes, similarity):
    return ResNet(BasicBlock, [200, 200, 200], num_classes, similarity)


def test(net):
    import numpy as np

    total_params = 0

    for x in filter(lambda p: p.requires_grad, net.parameters()):
        total_params += np.prod(x.data.numpy().shape)
    print("Total number of params", total_params)
    print(
        "Total layers",
        len(
            list(
                filter(
                    lambda p: p.requires_grad and len(p.data.size()) > 1,
                    net.parameters(),
                )
            )
        ),
    )
