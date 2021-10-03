import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import spectral_norm
from torch.nn import init
from .genOdinModel import euc_dist_layer, cosine_layer


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


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option="A"):
        super(BasicBlock, self).__init__()
        self.conv1 = spectral_norm(
            nn.Conv2d(
                in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
            )
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = spectral_norm(
            nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.activation = F.leaky_relu if self.mod else F.relu
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
                    spectral_norm(
                        nn.Conv2d(
                            in_planes,
                            self.expansion * planes,
                            kernel_size=1,
                            stride=stride,
                            bias=False,
                        )
                    ),
                    nn.BatchNorm2d(self.expansion * planes),
                )

    def forward(self, x):
        out = self.activation(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
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
        similarity="E",
        selfsupervision=False,
        batch_size=128,
    ):
        super(ResNet, self).__init__()
        self.in_planes = 16
        self.softmax = nn.Softmax(dim=-1)
        self.batch_size = batch_size
        self.do_not_genOdin = do_not_genOdin

        print("similarity: ", similarity)

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)

        if self.do_not_genOdin:
            self.outlayer = spectral_norm(nn.Linear(64, num_classes))
        else:

            self.similarity = similarity
            self.g_activation = nn.Sigmoid()
            self.g_func = nn.Linear(self.fc1.out_features, 1)
            self.g_norm = nn.BatchNorm1d(self.g_func.out_features)

            self.selfsupervision = selfsupervision
            if self.selfsupervision:
                self.x_trans_head = spectral_norm(nn.Linear(64, 3))
                self.y_trans_head = spectral_norm(nn.Linear(64, 3))
                self.rot_head = spectral_norm(nn.Linear(64, 4))
            self.pred_layer = spectral_norm(nn.Linear(num_classes, num_classes - 1))

            if self.similarity == "I":
                self.dropout_3 = nn.Dropout(p=0.4)
                self.h_func = spectral_norm(nn.Linear(64, num_classes))

            elif self.similarity == "E":
                self.dropout_3 = nn.Dropout(0)
                self.h_func = euc_dist_layer(num_classes, 64)

            elif self.similarity == "C":
                self.dropout_3 = nn.Dropout(p=0)
                self.h_func = cosine_layer(num_classes, 64)
            else:
                assert False, "Incorrect similarity Measure"

        self.apply(_weights_init)
        self.activation = F.leaky_relu if self.mod else F.relu

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x, get_test_model=False, train_g=False, self_sup_train=False):
        out = self.activation(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])

        out = out.view(out.size(0), -1)

        if self.do_not_genOdin:
            return self.softmax(self.outlayer(out))

        g = self.g_activation(self.g_norm(self.g_func(out)))

        if train_g:
            return g

        h = self.h_func(out)
        pred = self.softmax(torch.div(g, h))

        if self_sup_train:
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


def resnet20():
    return ResNet(BasicBlock, [3, 3, 3])


def resnet32():
    return ResNet(BasicBlock, [5, 5, 5])


def resnet44():
    return ResNet(BasicBlock, [7, 7, 7])


def resnet56():
    return ResNet(BasicBlock, [9, 9, 9])


def resnet110():
    return ResNet(BasicBlock, [18, 18, 18])


def resnet1202():
    return ResNet(BasicBlock, [200, 200, 200])