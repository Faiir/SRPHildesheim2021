import argparse

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms.functional as trnF
import torchvision.datasets as dset
import torch.nn.functional as F

import sys
import os


PACKAGE_PARENT = ".."
SCRIPT_DIR = os.path.dirname(
    os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__)))
)
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))


from project.model.get_model import add_rot_heads
from project.selfsupervision.pertubed_dataset import create_pert_dataloader

from project.model.get_model import get_model, save_model
from project.data.data_manager import get_data_manager

""" 
Code From: https://github.com/hendrycks/ss-ood
Title: Using Self-Supervised Learning Can Improve Model Robustness and Uncertainty
Author:Dan Hendrycks and Mantas Mazeika and Saurav Kadavath and Dawn Song
"""


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
        target_class = target_class.int()

        # target_class += 1
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
        )

        optimizer.zero_grad(set_to_none=True)

        # Forward together
        logits, x_trans_logits, y_trans_logits, rot_logits = net(
            batch, self_sup_train=True
        )

        # print("pen", pen.size())
        classification_logits = logits[:batch_size]
        # rot_logits = net.rot_head(pen[: 4 * batch_size])
        # x_trans_logits = net.x_trans_head(pen[4 * batch_size :])
        # y_trans_logits = net.y_trans_head(pen[4 * batch_size :])

        classification_loss = F.cross_entropy(
            classification_logits, target_class.cuda().long()
        )

        rot_loss = (
            F.cross_entropy(rot_logits, target_rots.cuda().long()) * rot_loss_weight
        )
        x_trans_loss = (
            F.cross_entropy(x_trans_logits, target_trans_x.cuda().long())
            * transl_loss_weight
        )
        y_trans_loss = (
            F.cross_entropy(y_trans_logits, target_trans_y.cuda().long())
            * transl_loss_weight
        )

        loss = classification_loss + ((rot_loss + x_trans_loss + y_trans_loss) / 3.0)

        loss.backward()

        optimizer.step()
        lr_scheduler.step()
        # exponential moving average
        loss_avg = loss_avg * 0.9 + float(loss) * 0.1
        print(loss_avg)
    return net


def ss_experiment(
    data_manager,
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
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    resnet20 = get_model(
        "small_gen_odin_res", similarity="C", selfsupervision=True, num_classes=11
    )
    resnet20 = add_rot_heads(resnet20)

    if torch.cuda.is_available():
        resnet20.cuda()

    ss_loader = create_pert_dataloader(data_manager, batchsize)

    optimizer = torch.optim.SGD(
        resnet20.parameters(),
        lr,
        momentum=momentum,
        weight_decay=weight_decay,
        nesterov=True,
    )

    lr_sheduler = create_lr_sheduler(optimizer, epochs, ss_loader, lr)

    resnet20 = train_ss(
        resnet20,
        ss_loader,
        optimizer,
        lr_sheduler,
        rot_loss_weight,
        transl_loss_weight,
    )

    if save_net:
        if kwargs["desc_string"] is not None:
            save_model(resnet20, kwargs["path"], kwargs["desc_string"])
        else:
            save_model(resnet20, kwargs["path"])
    return resnet20


if __name__ == "__main__":
    #!python -m ss_experiment --batchsize 128 --epochs 1 --lr 0.0001 --momentum 0.8 --weight_decay 0.00001 --rot_loss_weight 1.0 --trans_loss_weight 1.0 --oodratio 0.3 --poolsize 20000 --trainsize 1500 --oo_dist MNIST --oo_dist Fashion_MNIST
    parser = argparse.ArgumentParser("argument for training")

    parser.add_argument("--batchsize", type=int, default=256, help="batchsize")
    parser.add_argument("--epochs", type=int, default=50, help="save frequency")
    parser.add_argument("--lr", type=float, default=0.0001, help="learning rate")
    parser.add_argument("--momentum", type=float, default=0.7, help="momentum")
    parser.add_argument(
        "--weight_decay", type=float, default=0.0001, help="weight_decay"
    )
    parser.add_argument(
        "--rot_loss_weight", type=float, default=0.5, help="rot_loss_weight"
    )
    parser.add_argument(
        "--trans_loss_weight", type=float, default=0.5, help="transl_loss_weight"
    )
    parser.add_argument(
        "--output", type=str, default=0.5, help=r"..\model\saved_models"
    )
    parser.add_argument("--in_dist", type=str, default="Cifar10", help=r"Cifar10")
    parser.add_argument(
        "--oo_dist", nargs="+", type=str, default="MNIST", help=r"OOD datasets"
    )
    parser.add_argument("--oodratio", type=float, default=0.2, help=r"% of ood sample")
    parser.add_argument("--poolsize", type=int, default=250000, help="pool_size")
    parser.add_argument("--trainsize", type=int, default=2000, help="train_size")
    opt = parser.parse_args()

    data_manager = get_data_manager([opt.in_dist], ood=opt.oo_dist)
    data_manager.create_merged_data(
        test_size=opt.trainsize,
        pool_size=opt.poolsize,
        labelled_size=opt.trainsize,
        OOD_ratio=opt.oodratio,
    )
    ss_experiment(
        data_manager,
        opt.batchsize,
        opt.epochs,
        opt.lr,
        opt.momentum,
        opt.weight_decay,
        opt.rot_loss_weight,
        opt.trans_loss_weight,
        True,
        path=opt.output,
    )
