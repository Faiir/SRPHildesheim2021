import numpy as np
import torch
from tqdm import tqdm

import torch.nn as nn
import torchvision.transforms.functional as trnF
import torchvision.datasets as dset
import torch.nn.functional as F
from .pertubed_dataset import create_pert_dataloader
from ..model.resnet import add_rot_heads
from ..model.get_model import get_model, save_model


def cosine_annealing(step, total_steps, lr_max, lr_min):
    return lr_min + (lr_max - lr_min) * 0.5 * (1 + np.cos(step / total_steps * np.pi))


def create_lr_sheduler(optimizer, epochs, pert_loader, learning_rate):

    return torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: cosine_annealing(
            step,
            epochs * len(pert_loader),
            1,  # since lr_lambda computes multiplicative factor
            1e-6 / learning_rate,
        ),
    )


def train_ss(
    net,
    train_loader_in,
    optimizer,
    lr_scheduler,
    rot_loss_weight,
    transl_loss_weight,
):
    net.train()  # enter train mode
    loss_avg = 0.0
    for (
        x_tf_0,
        x_tf_90,
        x_tf_180,
        x_tf_270,
        x_tf_trans,
        target_trans_x,
        target_trans_y,
        target_class,
    ) in tqdm(train_loader_in, dynamic_ncols=True):
        batch_size = x_tf_0.shape[0]

        # Sanity check
        assert (
            x_tf_0.shape[0]
            == x_tf_90.shape[0]
            == x_tf_180.shape[0]
            == x_tf_270.shape[0]
            == x_tf_trans.shape[0]
            == target_trans_x.shape[0]
            == target_trans_y.shape[0]
            == target_class.shape[0]
        )

        batch = np.concatenate((x_tf_0, x_tf_90, x_tf_180, x_tf_270, x_tf_trans), 0)
        batch = torch.FloatTensor(batch).cuda()

        target_rots = torch.cat(
            (
                torch.zeros(batch_size),
                torch.ones(batch_size),
                2 * torch.ones(batch_size),
                3 * torch.ones(batch_size),
            ),
            0,
        ).long()

        lr_scheduler.step()
        optimizer.zero_grad()

        # Forward together
        logits, pen = net(batch, self_supervision=True)

        classification_logits = logits[:batch_size]
        rot_logits = net.rot_head(pen[: 4 * batch_size])
        x_trans_logits = net.x_trans_head(pen[4 * batch_size :])
        y_trans_logits = net.y_trans_head(pen[4 * batch_size :])

        classification_loss = F.cross_entropy(
            classification_logits, target_class.cuda()
        )
        rot_loss = F.cross_entropy(rot_logits, target_rots.cuda()) * rot_loss_weight
        x_trans_loss = (
            F.cross_entropy(x_trans_logits, target_trans_x.cuda()) * transl_loss_weight
        )
        y_trans_loss = (
            F.cross_entropy(y_trans_logits, target_trans_y.cuda()) * transl_loss_weight
        )

        loss = classification_loss + ((rot_loss + x_trans_loss + y_trans_loss) / 3.0)

        loss.backward()
        optimizer.step()

        # exponential moving average
        loss_avg = loss_avg * 0.9 + float(loss) * 0.1
        print(loss_avg)
    return net


def ss_experiment(
    datamanager,
    batchsize,
    epochs=50,
    lr=0.001,
    momentum=0.7,
    weight_decay=0.0001,
    rot_loss_weight=0.5,
    transl_loss_weight=0.5,
    save_net=True,
    **kwargs
):
    resnet18 = get_model("gen_odin_res", similarity="C")
    resnet18 = add_rot_heads(resnet18)

    ss_loader = create_pert_dataloader(datamanager, batchsize)

    optimizer = torch.optim.SGD(
        resnet18.parameters(),
        lr,
        momentum=momentum,
        weight_decay=weight_decay,
        nesterov=True,
    )

    lr_sheduler = create_lr_sheduler(optimizer, epochs, ss_loader, lr)

    resnet18 = train_ss(
        resnet18,
        ss_loader,
        optimizer,
        lr_sheduler,
        rot_loss_weight,
        transl_loss_weight,
    )
    if save_net:
        save_model(resnet18, kwargs["path"], kwargs["desc_string"])

    return resnet18
