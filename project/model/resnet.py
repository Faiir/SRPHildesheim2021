"""
Pytorch implementation of ResNet models.
Reference:
[1] He, K., Zhang, X., Ren, S., Sun, J.: Deep residual learning for image recognition. In: CVPR, 2016.
"""
import torch
import math
import torch.nn as nn
import torch.nn.functional as F

from .spectral_normalization.spectral_norm_conv_inplace import spectral_norm_conv
from .spectral_normalization.spectral_norm_fc import spectral_norm_fc
from .genOdinModel import cosine_layer, euc_dist_layer


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


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, input_size, wrapped_conv, in_planes, planes, stride=1, mod=True):
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
        out = self.activation(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.activation(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, input_size, wrapped_conv, in_planes, planes, stride=1, mod=True):
        super(Bottleneck, self).__init__()
        self.conv1 = wrapped_conv(
            input_size, in_planes, planes, kernel_size=1, stride=1
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = wrapped_conv(
            input_size, planes, planes, kernel_size=3, stride=stride
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = wrapped_conv(
            math.ceil(input_size / stride),
            planes,
            self.expansion * planes,
            kernel_size=1,
            stride=1,
        )
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)
        self.mod = mod
        self.activation = F.leaky_relu if self.mod else F.relu

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
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
                    nn.BatchNorm2d(self.expansion * planes),
                )

    def forward(self, x):
        out = self.activation(self.bn1(self.conv1(x)))
        out = self.activation(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = self.activation(out)
        return out


class ResNet(nn.Module):
    def __init__(
        self,
        block,
        num_blocks,
        num_classes=10,
        temp=1.0,
        spectral_normalization=True,
        mod=True,
        coeff=3,
        n_power_iterations=1,
        mnist=False,
        similarity="E",
        selfsupervision=False,
        batch_size=128,
        do_not_genOdin=True,
    ):
        """
        If the "mod" parameter is set to True, the architecture uses 2 modifications:
        1. LeakyReLU instead of normal ReLU
        2. Average Pooling on the residual connections.
        """
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.softmax = nn.Softmax(dim=-1)
        self.mod = mod
        self.batch_size = batch_size
        self.do_not_genOdin = do_not_genOdin

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

        self.bn1 = nn.BatchNorm2d(64)

        if mnist:
            self.conv1 = wrapped_conv(28, 1, 64, kernel_size=3, stride=1)
            self.layer1 = self._make_layer(block, 28, 64, num_blocks[0], stride=1)
            self.layer2 = self._make_layer(block, 28, 128, num_blocks[1], stride=2)
            self.layer3 = self._make_layer(block, 14, 256, num_blocks[2], stride=2)
            self.layer4 = self._make_layer(block, 7, 512, num_blocks[3], stride=2)
        else:
            self.conv1 = wrapped_conv(32, 3, 64, kernel_size=3, stride=1)
            self.layer1 = self._make_layer(block, 32, 64, num_blocks[0], stride=1)
            self.layer2 = self._make_layer(block, 32, 128, num_blocks[1], stride=2)
            self.layer3 = self._make_layer(block, 16, 256, num_blocks[2], stride=2)
            self.layer4 = self._make_layer(block, 8, 512, num_blocks[3], stride=2)
        self.fc = nn.Linear(512 * block.expansion, 128)
        self.fc1 = nn.Linear(128, num_classes)

        if self.do_not_genOdin:
            outlayer = nn.Linear(self.fc1.out_features, num_classes)

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


def resnet18(
    spectral_normalization=True,
    mod=True,
    temp=1.0,
    mnist=False,
    similarity="C",
    **kwargs
):
    similarity = similarity
    model = ResNet(
        BasicBlock,
        [2, 2, 2, 2],
        spectral_normalization=spectral_normalization,
        mod=mod,
        temp=temp,
        mnist=mnist,
        similarity=similarity,
        **kwargs,
    )
    return model


def resnet50(spectral_normalization=True, mod=True, temp=1.0, mnist=False, **kwargs):
    model = ResNet(
        Bottleneck,
        [3, 4, 6, 3],
        spectral_normalization=spectral_normalization,
        mod=mod,
        temp=temp,
        mnist=mnist,
        **kwargs,
    )
    return model


def resnet101(spectral_normalization=True, mod=True, temp=1.0, mnist=False, **kwargs):
    model = ResNet(
        Bottleneck,
        [3, 4, 23, 3],
        spectral_normalization=spectral_normalization,
        mod=mod,
        temp=temp,
        mnist=mnist,
        **kwargs,
    )
    return model


def resnet110(spectral_normalization=True, mod=True, temp=1.0, mnist=False, **kwargs):
    model = ResNet(
        Bottleneck,
        [3, 4, 26, 3],
        spectral_normalization=spectral_normalization,
        mod=mod,
        temp=temp,
        mnist=mnist,
        **kwargs,
    )
    return model


def resnet152(spectral_normalization=True, mod=True, temp=1.0, mnist=False, **kwargs):
    model = ResNet(
        Bottleneck,
        [3, 8, 36, 3],
        spectral_normalization=spectral_normalization,
        mod=mod,
        temp=temp,
        mnist=mnist,
        **kwargs,
    )
    return model
