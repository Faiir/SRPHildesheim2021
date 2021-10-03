import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init


""" 
Github with code:
Author: Yerlan Idelbayev
Title: Proper ResNet Implementation for CIFAR10/CIFAR100 in Pytorch
Github Link: https://github.com/akamaster/pytorch_resnet_cifar10


Resnet
Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
[2] https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
"""

from .spectral_normalization.spectral_norm_conv_inplace import spectral_norm_conv
from .spectral_normalization.spectral_norm_fc import spectral_norm_fc
from .genOdinModel import cosine_layer, euc_dist_layer
import math


class AvgPoolShortCut(nn.Module):
    def __init__(self, stride, out_c, in_c):
        super(AvgPoolShortCut, self).__init__()
        self.stride = stride
        self.out_c = out_c
        self.in_c = in_c

    def forward(self, x):
        if x.shape[2] % 2 != 0:
            x = F.avg_pool2d(x, 1, self.stride)
        else:
            x = F.avg_pool2d(x, self.stride, self.stride)
        pad = torch.zeros(
            x.shape[0],
            self.out_c - self.in_c,
            x.shape[2],
            x.shape[3],
            device=x.device,
        )
        x = torch.cat((x, pad), dim=1)
        return x


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


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        input_size,
        wrapped_conv,
        in_planes,
        planes,
        stride=1,
        mod=True,
        option="B",
    ):
        super(BasicBlock, self).__init__()
        self.conv1 = wrapped_conv(
            input_size, in_planes, planes, kernel_size=3, stride=stride
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = wrapped_conv(
            math.ceil(input_size / stride), planes, planes, kernel_size=3, stride=1
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.mod = mod
        self.activation = F.leaky_relu if self.mod else F.relu

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            if option == "A":  # TODO
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
                if mod:
                    self.shortcut = nn.Sequential(
                        AvgPoolShortCut(stride, self.expansion * planes, in_planes)
                    )
                else:
                    self.shortcut = nn.Sequential(
                        wrapped_conv(
                            input_size,
                            in_planes,
                            self.expansion * planes,
                            kernel_size=1,
                            stride=stride,
                        ),
                        nn.BatchNorm2d(planes),
                    )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(
        self,
        block,
        num_blocks,
        do_not_genOdin,
        num_classes=10,
        temp=1.0,
        spectral_normalization=True,
        mod=True,
        coeff=3,
        n_power_iterations=1,
        similarity="E",
        selfsupervision=False,
        batch_size=128,
    ):
        super(ResNet, self).__init__()
        self.in_planes = 16
        self.softmax = nn.Softmax(dim=-1)
        self.mod = mod
        self.batch_size = batch_size
        self.do_not_genOdin = do_not_genOdin
        print("similarity: ", similarity)

        def wrapped_conv(input_size, in_c, out_c, kernel_size, stride):
            padding = 1 if kernel_size == 3 else 0

            conv = nn.Conv2d(in_c, out_c, kernel_size, stride, padding, bias=False)

            if not spectral_normalization:
                return conv

            # NOTE: Google uses the spectral_norm_fc in all cases
            if kernel_size == 1:
                # use spectral norm fc, because bound are tight for 1x1 convolutions
                wrapped_conv = spectral_norm_fc(conv, coeff, n_power_iterations)
            else:
                # Otherwise use spectral norm conv, with loose bound
                shapes = (in_c, input_size, input_size)
                wrapped_conv = spectral_norm_conv(
                    conv, coeff, shapes, n_power_iterations
                )

            return wrapped_conv

        self.wrapped_conv = wrapped_conv

        self.conv1 = wrapped_conv(32, 3, 16, kernel_size=3, stride=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 32, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 16, 64, num_blocks[2], stride=2)
        self.fc = nn.Linear(256 * block.expansion, 128)
        self.fc1 = nn.Linear(128, num_classes)

        if self.do_not_genOdin:
            self.outlayer = nn.Linear(self.fc1.out_features, num_classes)
        else:

            self.similarity = similarity
            self.g_activation = nn.Sigmoid()
            self.g_func = nn.Linear(self.fc1.out_features, 1)
            self.g_norm = nn.BatchNorm1d(self.g_func.out_features)

            self.selfsupervision = selfsupervision
            if self.selfsupervision:
                self.x_trans_head = nn.Linear(self.fc.out_features, 3)
                self.y_trans_head = nn.Linear(self.fc.out_features, 3)
                self.rot_head = nn.Linear(self.fc.out_features, 4)
            self.pred_layer = nn.Linear(num_classes, num_classes - 1)

            if self.similarity == "I":
                self.dropout_3 = nn.Dropout(p=0.6)
                self.h_func = nn.Linear(self.fc1.out_features, num_classes)

            elif self.similarity == "E":
                self.dropout_3 = nn.Dropout(0)
                self.h_func = euc_dist_layer(num_classes, self.fc1.out_features)

            elif self.similarity == "C":
                self.dropout_3 = nn.Dropout(p=0)
                self.h_func = cosine_layer(num_classes, self.fc1.out_features)
            else:
                assert False, "Incorrect similarity Measure"

        self.activation = F.leaky_relu if self.mod else F.relu
        self.feature = None
        self.temp = temp  #! change
        self.apply(_weights_init)

        self.apply(_weights_init)

    def _make_layer(self, block, input_size, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(
                block(
                    input_size,
                    self.wrapped_conv,
                    self.in_planes,
                    planes,
                    stride,
                    self.mod,
                )
            )
            self.in_planes = planes * block.expansion
            input_size = math.ceil(input_size / stride)
        return nn.Sequential(*layers)

    def forward(self, x, get_test_model=False, train_g=False, self_sup_train=False):

        out = self.activation(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        # print("flatten", out.size())  # 4x256
        out = self.activation(self.fc(out))
        f_out = self.fc1(out)
        if self.do_not_genOdin:
            return self.softmax(self.outlayer(f_out))
        # if self_supervision:
        #     return f_out, out  # (128,10) (n,128)

        self.feature = out.clone().detach()
        # out = self.fc(out) / self.temp
        g = self.g_activation(self.g_norm(self.g_func(f_out)))
        if train_g:
            return g

        h = self.h_func(f_out)
        pred = self.softmax(torch.div(g, h))  # 128 11

        if self_sup_train:
            # x_trans = self.x_trans_head(out)
            # y_trans = self.y_trans_head(out)
            # rot = self.rot_head(out)
            x_trans = self.x_trans_head(out[4 * self.batch_size :])
            y_trans = self.y_trans_head(out[4 * self.batch_size :])  # 128 3
            rot = self.rot_head(out[: 4 * self.batch_size])  # 512 4
            return pred, x_trans, y_trans, rot

        if self.selfsupervision:
            return self.softmax(pred_layer(pred))

        if not get_test_model:
            return pred
        else:
            return pred, g, h


def resnet20(spectral_normalization=True, mod=True, temp=1.0, similarity="C", **kwargs):
    return ResNet(
        BasicBlock,
        num_blocks=[3, 3, 3],
        spectral_normalization=spectral_normalization,
        mod=mod,
        temp=temp,
        similarity=similarity,
        num_classes=kwargs.get("num_classes"),
        selfsupervision=kwargs.get("selfsupervision"),
        do_not_genOdin=kwargs.get("do_not_genOdin", False),
    )


def resnet32(
    spectral_normalization=True,
    mod=True,
    temp=1.0,
    mnist=False,
    similarity="C",
    **kwargs
):
    return ResNet(
        BasicBlock,
        [5, 5, 5],
        spectral_normalization=spectral_normalization,
        mod=mod,
        temp=temp,
        mnist=mnist,
        **kwargs,
    )


def resnet44(
    spectral_normalization=True,
    mod=True,
    temp=1.0,
    mnist=False,
    similarity="C",
    **kwargs
):
    return ResNet(
        BasicBlock,
        [7, 7, 7],
        spectral_normalization=spectral_normalization,
        mod=mod,
        temp=temp,
        mnist=mnist,
        **kwargs,
    )


def resnet56(
    spectral_normalization=True,
    mod=True,
    temp=1.0,
    mnist=False,
    similarity="C",
    **kwargs
):
    return ResNet(
        BasicBlock,
        [9, 9, 9],
        spectral_normalization=spectral_normalization,
        mod=mod,
        temp=temp,
        mnist=mnist,
        **kwargs,
    )


def resnet110(
    spectral_normalization=True,
    mod=True,
    temp=1.0,
    mnist=False,
    similarity="C",
    **kwargs
):
    return ResNet(
        BasicBlock,
        [18, 18, 18],
        spectral_normalization=spectral_normalization,
        mod=mod,
        temp=temp,
        mnist=mnist,
        **kwargs,
    )


def resnet1202(
    spectral_normalization=True,
    mod=True,
    temp=1.0,
    mnist=False,
    similarity="C",
    **kwargs
):
    return ResNet(
        BasicBlock,
        [200, 200, 200],
        spectral_normalization=spectral_normalization,
        mod=mod,
        temp=temp,
        mnist=mnist,
        **kwargs,
    )


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


def add_rot_heads(net, pernumile_layer_size=64):
    net.x_trans_head = nn.Linear(pernumile_layer_size, 3)
    net.y_trans_head = nn.Linear(pernumile_layer_size, 3)
    net.rot_head = nn.Linear(pernumile_layer_size, 4)

    return net