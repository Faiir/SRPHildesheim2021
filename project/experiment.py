# torch
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.utils.tensorboard import SummaryWriter

# python imports
from datetime import datetime
import os
from tqdm import tqdm
import json
import pandas as pd


# data imports
from .data.datahandler_for_array import get_dataloader
from .data.datamanager import get_datamanager

# train functions
from .model.train import train, test

from .model.get_model import get_model


# helpers
from .helpers.measures import accuracy, f1, auroc
from .helpers.get_pool_predictions import get_pool_predictions

from .helpers.get_tsne_plot import get_tsne_plot

do_tsne = False


def experiment(param_dict, oracle, data_manager, writer, dataset, net, verbose=0):
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

    oracle_stepsize = param_dict["oracle_stepsize"]
    oracle_steps = param_dict["oracle_steps"]
    epochs = param_dict["epochs"]
    batch_size = param_dict["batch_size"]
    weight_decay = param_dict["weight_decay"]
    metric = param_dict["metric"]

    if oracle == "random":
        from .helpers.sampler import random_sample

        sampler = random_sample
    elif oracle == "highest entropy":
        from .helpers.sampler import uncertainity_sampling_highest_entropy

        sampler = uncertainity_sampling_highest_entropy
    elif oracle == "least confidence":
        from .helpers.sampler import uncertainity_sampling_least_confident

        sampler = uncertainity_sampling_least_confident
    elif oracle == "DDU":
        from .helpers.sampler import DDU_sampler

        sampler = DDU_sampler
    elif oracle == "Gen0din":
        from .helpers.sampler import gen0din_sampler

        sampler = gen0din_sampler

    # net = torch.hub.load('pytorch/vision:v0.9.0', 'resnet18', pretrained=False)
    # net = get_model("base")  # torchvision.models.resnet18(pretrained=False)
    if torch.cuda.is_available():
        net.cuda()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    data_manager.reset_pool()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    for i in tqdm(range(oracle_steps)):
        if do_tsne:
            tsne_plot = get_tsne_plot(data_manager, dataset, net, device)
            writer.add_figure(
                tag=f"{metric}/{dataset}/{oracle}/tsne{i}", figure=tsne_plot
            )

        train_loader, test_loader, pool_loader = get_dataloader(
            data_manager, batch_size=batch_size
        )

        if torch.cuda.is_available():
            net.cuda()
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(net.parameters(), weight_decay=weight_decay)

        trained_net, avg_train_loss = train(
            net,
            train_loader,
            optimizer,
            criterion,
            device=device,
            epochs=epochs,
            verbose=verbose,
        )

        avg_test_loss = test(
            trained_net, criterion, test_loader, device=device, verbose=verbose
        )

        avg_test_loss = test(
            trained_net, criterion, test_loader, device=device, verbose=verbose
        )

        # unlabelled pool predictions
        pool_predictions, pool_labels_list = get_pool_predictions(
            trained_net, pool_loader, device=device, return_labels=True
        )

        # samples from unlabelled pool predictions
        sampler(
            dataset_manager=data_manager,
            number_samples=oracle_stepsize,
            net=trained_net,
            predictions=pool_predictions,
        )

        test_predictions, test_labels = get_pool_predictions(
            trained_net, test_loader, device=device, return_labels=True
        )
        train_predictions, train_labels = get_pool_predictions(
            trained_net, train_loader, device=device, return_labels=True
        )

        if metric.lower() == "accuracy":
            test_accuracy = accuracy(test_labels, test_predictions)
            train_accuracy = accuracy(train_labels, train_predictions)

            dict_to_add = {
                "test_loss": avg_test_loss,
                "train_loss": avg_train_loss,
                "test_accuracy": test_accuracy,
                "train_accuracy": train_accuracy,
            }

        elif metric.lower() == "f1":
            f1_score = f1(test_labels, test_predictions)
            dict_to_add = {"f1": f1_score}
        elif metric.lower() == "auroc":
            auroc_score = auroc(data_manager, i)

            dict_to_add = {"auroc": auroc_score}

        data_manager.add_log(
            writer=writer,
            oracle=oracle,
            dataset=dataset,
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


def start_experiment(config_path, log):
    """start_experiment [function which starts all experiments in the config json - main function of this module]

    [extended_summary]

    Args:
        config_path ([String]): [path to the json config]
        log ([String]): [path to log folder]
    """

    # print(config_path)
    # print(log)
    log_dir = log
    config = ""

    config_path = os.path.join(config_path)

    writer = SummaryWriter()

    with open(config_path, mode="r", encoding="utf-8") as config_f:
        config = json.load(config_f)

    datasets = config["datasets"]
    for dataset in datasets:

        data_manager = get_datamanager(
            indistribution=["Cifar10"], ood=["Fashion_MNIST"]
        )

        for exp in config["experiment-list"]:
            metric = exp["metric"]

            for oracle in exp["oracles"]:

                net = get_model(
                    exp["model_name"],
                    similarity=exp["similarity"],
                    out_classes=10,
                    include_bn=False,
                    channel_input=3,
                )

                data_manager.create_merged_data(
                    test_size=exp["test_size"],
                    pool_size=exp["pool_size"],
                    labelled_size=exp["labelled_size"],
                    OOD_ratio=exp["OOD_ratio"],
                )

                trained_net = experiment(
                    param_dict=exp,
                    oracle=oracle,
                    data_manager=data_manager,
                    writer=writer,
                    dataset=dataset,
                    verbose=0,
                    net=net,
                )

                log_df = data_manager.get_logs()

                current_time = datetime.now().strftime("%H-%M-%S")
                log_file_name = "Experiment-from-" + str(current_time) + ".csv"

                log_dir = os.path.join(".", "log_dir")

                if os.path.exists(log_dir) == False:
                    os.mkdir(os.path.join(".", "log_dir"))

                log_path = os.path.join(log_dir, log_file_name)

                with open(log_path, mode="w", encoding="utf-8") as logfile:
                    colums = log_df.columns
                    for colum in colums:
                        logfile.write(colum + ",")
                    logfile.write("\n")
                    for _, row in log_df.iterrows():
                        for c in colums:
                            logfile.write(str(row[c].item()))
                            logfile.write(",")
                        logfile.write("\n")
    print(
        """
    **********************************************


                  EXPERIMENT DONE

    **********************************************
    """
    )
