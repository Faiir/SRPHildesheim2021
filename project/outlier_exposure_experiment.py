from project.helpers.sampler import (
    uncertainity_sampling_highest_entropy,
)
from project.model.train import train, test, get_density_vals
from project.helpers.measures import accuracy
from project.helpers.get_pool_predictions import get_pool_predictions
from project.helpers.get_density_plot import density_plot
from project.model.train import train_g

# torch
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision

# python imports
from datetime import datetime
import os
from tqdm import tqdm
import json
import pandas as pd
from tqdm import tqdm
from project.data.datahandler_for_array import get_dataloader


def outlier_exposure_experiment(data_manager, writer, net, verbose=0, **kwargs):
    """experiment [Experiment function which performs the entire acitve learning process based on the predefined config]

    [extended_summary]

    Args:
        param_dict ([dict]): [experiment config from json file]
        data_manager ([class]): [Data manager which handels the management of both the dataset and the OOD data. Logs, Samples, performs oracle steps etc.]
        net ([nn.module]): [Pytorch Neural Network for experiment]
        verbose (int, optional): [description]. Defaults to 0.

    Returns:
        [None]: [Log Attribute in Datamanage writes log to dist]
    """

    oracle_stepsize = kwargs.get("oracle_stepsize", 200)
    oracle_steps = kwargs.get("oracle_steps", 10)
    epochs = kwargs.get("epochs", 25)
    batch_size = kwargs.get("batch_size", 64)
    weight_decay = kwargs.get("weight_decay", 0.0001)
    metric = "accuracy"
    momentum = kwargs.get("momentum", 0.8)
    outlier_exposure_amount = kwargs.get("outlier_exposure_amount", 2)
    lr = kwargs.get("lr", 0.0001)

    sampler = uncertainity_sampling_highest_entropy

    # net = torch.hub.load('pytorch/vision:v0.9.0', 'resnet18', pretrained=False)
    # net = get_model("base")  # torchvision.models.resnet18(pretrained=False)
    if torch.cuda.is_available():
        net.cuda()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    data_manager.reset_pool()

    for i in tqdm(range(oracle_steps)):

        train_loader, test_loader, pool_loader = get_dataloader(
            data_manager, batch_size=batch_size
        )

        if torch.cuda.is_available():
            net.cuda()
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(
            net.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay
        )

        if (i % outlier_exposure_amount == 3) and (i != 0):
            # init  optimizer
            g_optim = optim.SGD(
                net.parameters(),
                lr=lr * 5,
                momentum=momentum,
                weight_decay=weight_decay,
            )
            net = train_g(net, g_optim, data_manager, epochs=10)

        net, avg_train_loss = train(
            net,
            train_loader,
            optimizer,
            criterion,
            device=device,
            epochs=epochs,
            verbose=verbose,
        )
        net.eval()
        avg_test_loss = test(
            net, criterion, test_loader, device=device, verbose=verbose
        )
        print(f"avg.test loss: {avg_test_loss} oracle step {i}")
        pert_preds, gs, hs, targets = get_density_vals(pool_loader, test_loader, net)
        density_plot(pert_preds, gs, hs, targets, writer, i)
        # unlabelled pool predictions
        pool_predictions, pool_labels_list = get_pool_predictions(
            net, pool_loader, device=device, return_labels=True
        )

        # samples from unlabelled pool predictions
        sampler(
            dataset_manager=data_manager,
            number_samples=oracle_stepsize,
            net=net,
            predictions=pool_predictions,
        )

        test_predictions, test_labels = get_pool_predictions(
            net, test_loader, device=device, return_labels=True
        )
        train_predictions, train_labels = get_pool_predictions(
            net, train_loader, device=device, return_labels=True
        )

        if metric.lower() == "accuracy":
            test_accuracy = accuracy(test_labels, test_predictions)
            train_accuracy = accuracy(train_labels, train_predictions)
            print(f"train_accuracy: {train_accuracy} oracle step {i}")
            dict_to_add = {
                "test_loss": avg_test_loss,
                "train_loss": avg_train_loss,
                "test_accuracy": test_accuracy,
                "train_accuracy": train_accuracy,
            }

        data_manager.add_log(
            writer=writer,
            oracle="entropy",
            dataset="cifar-fmnist",
            metric=metric,
            log_dict=dict_to_add,
        )

        # data_manager.add_log(log_dict=dict_to_add)
        # print(dict_to_add)

        # if oracle is not None:
        #     predictions = get_pool_predictions(trained_net, pool_loader, device=device)
        #     oracle(
        #         dataset_manager=data_manager,
        #         number_samples=oracle_stepsize,
        #         predictions=predictions,
        #     )

    return net