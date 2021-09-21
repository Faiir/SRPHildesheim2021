import argparse
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms.functional as trnF
import torchvision.datasets as dset
import torch.nn.functional as F
from .pertubed_dataset import create_pert_dataloader
from ..model.resnet import add_rot_heads
from ..model.get_model import get_model, save_model
from ..data.datamanager import get_datamanager


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


if __name__ == "__main__":
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
        "--transl_loss_weight", type=float, default=0.5, help="transl_loss_weight"
    )
    parser.add_argument(
        "--output", type=str, default=0.5, help=r"..\model\saved_models"
    )
    parser.add_argument("--in_dist", type=str, default="Cifar10", help=r"Cifar10")
    parser.add_argument(
        "--oo_dist", nargs="+", type=str, default="MNIST", help=r"OOD datasets"
    )
    parser.add_argument("--ood_ratio", type=float, default=0.2, help=r"% of ood sample")
    parser.add_argument("--pool_size", type=int, default=250000, help="pool_size")
    parser.add_argument("--train_size", type=int, default=2000, help="train_size")
    opt = parser.parse_args()

    datamanager = get_datamanager([opt["in_dist"]], ood=opt["ood_ratio"])
    datamanager.create_merged_data(
        test_size=opt["train_size"],
        pool_size=opt["pool_size"],
        labelled_size=opt["train_size"],
        OOD_ratio=opt["ood_ratio"],
    )
    ss_experiment(
        datamanager,
        opt["batchsize"],
        opt["epochs"],
        opt["lr"],
        opt["momentum"],
        opt["weight_decay"],
        opt["rot_loss_weight"],
        opt["transl_loss_weight"],
        True,
        opt["output"],
    )
