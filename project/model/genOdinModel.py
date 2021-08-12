import torch.nn as nn
import torch.nn.functional as F
import torch


class euc_dist_layer(nn.Module):
    def __init__(self, out_classes):
        super().__init__()
        self.out_classes = torch.arange(out_classes)

    def forward(self, x):
        # https://pytorch.org/docs/stable/generated/torch.linalg.norm.html#torch.linalg.norm
        pass


class cosine_layer(nn.Module):
    def __init__(self, out_classes):
        super().__init__()
        self.out_classes = out_classes

    def forward(self, x):
        # https://pytorch.org/docs/stable/generated/torch.nn.CosineSimilarity.html
        pass


class genOdinModel(nn.module):
    def __init__(self):
        pass


class genOdinModel(nn.Module):
    """Net [Basic conv net for MNIST Dataset]"""

    def __init__(self, activation=F.relu, similarity="C"):
        super(genOdinModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 4, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(4, 12, 5)
        self.fc1 = nn.Linear(12 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 64)
        self.fc3 = nn.Linear(64, 10)
        self.dropout1 = nn.Dropout(p=0.4)
        self.dropout2 = nn.Dropout(p=0.4)

        self.similarity = similarity
        self.g_norm = nn.BatchNorm2d()
        self.g_activation = F.sigmoid()
        self.g_func = nn.Linear(1)

        self.activation = activation

        if similarity == "I":
            self.dropout_3 = nn.Dropout(p=0.6)
            self.h_func = nn.Linear(10)

        elif similarity == "E":
            self.dropout_3 = nn.Dropout(0)
            self.h_func = 0  # eulid_dist(10)

        elif similarity == "C":
            self.dropout_3 = nn.Dropout(p=0)
            self.h_func = 0  # cosine(10)
        else:
            assert False, "Incorrect similarity Measure"

    def forward(self, input_data, get_test_model=False):
        x = self.pool(self.activation(self.conv1(input_data)))
        x = self.pool(self.activation(self.conv2(x)))
        x = x.view(-1, 12 * 4 * 4)
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.fc3(x)
        x = F.softmax(x, dim=-1)
        if get_test_model:
            return x
        else:
            # return x, input_data, g_val, h_val
            pass
