import numpy as np
import os
import torch
import torch.nn as nn
import torchvision.transforms as trn
import torchvision.transforms.functional as trnF
import torchvision.datasets as dset
from torch.utils.data import DataLoader
import torch.nn.functional as F
from PIL import Image
import itertools
import cv2
import random
from PIL import Image, ImageOps, ImageEnhance, PILLOW_VERSION
import math

from .cv2utils import affine


normalize = trn.Normalize([0.5] * 3, [0.5] * 3)
randomly_crop = trn.RandomCrop(32, padding=4)


class PerturbDataset(torch.utils.data.Dataset):
    def __init__(self, data, targets, train_mode=True):
        self.data = data
        self.targets = targets
        self.num_points = len(self.data)
        self.train_mode = train_mode

    def __getitem__(self, index):
        x_orig = self.data[index]
        classifier_target = self.targets[index]

        if self.train_mode == True and np.random.uniform() < 0.5:
            x_orig = np.copy(x_orig)[:, ::-1]
        else:
            x_orig = np.copy(x_orig)

        if self.train_mode == True:
            # x_orig = Image.fromarray(x_orig)
            x_orig = randomly_crop(x_orig)
            x_orig = np.asarray(x_orig)

        x_tf_0 = np.copy(x_orig)
        x_tf_90 = np.rot90(x_orig.copy(), k=1).copy()
        x_tf_180 = np.rot90(x_orig.copy(), k=2).copy()
        x_tf_270 = np.rot90(x_orig.copy(), k=3).copy()

        possible_translations = list(itertools.product([0, 8, -8], [0, 8, -8]))
        num_possible_translations = len(possible_translations)
        tx, ty = possible_translations[random.randint(0, num_possible_translations - 1)]
        tx_target = {0: 0, 8: 1, -8: 2}[tx]
        ty_target = {0: 0, 8: 1, -8: 2}[ty]
        x_tf_trans = affine(
            np.asarray(x_orig).copy(),
            0,
            (tx, ty),
            1,
            0,
            interpolation=cv2.INTER_CUBIC,
            mode=cv2.BORDER_REFLECT_101,
        )

        return (
            normalize(trnF.to_tensor(x_tf_0)),
            normalize(trnF.to_tensor(x_tf_90)),
            normalize(trnF.to_tensor(x_tf_180)),
            normalize(trnF.to_tensor(x_tf_270)),
            normalize(trnF.to_tensor(x_tf_trans)),
            torch.tensor(tx_target),
            torch.tensor(ty_target),
            torch.tensor(classifier_target),
        )

    def __len__(self):
        return self.num_points


def create_pert_dataloader(datamanager, batchsize):
    pooldata, pooltarget = datamanager.get_unlabelled_pool_data()

    p_data = PerturbDataset(pooldata, pooltarget, train_mode=True)

    return torch.utils.data.DataLoader(
        p_data, batch_size=batchsize, shuffle=True, num_workers=2, pin_memory=False
    )
