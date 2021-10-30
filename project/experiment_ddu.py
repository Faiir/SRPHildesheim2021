# torch
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils import data
from torch.utils.tensorboard import SummaryWriter

# python imports
from datetime import datetime
import os
from tqdm import tqdm
import json
import pandas as pd
import gc

# data imports
from .data.datahandler_for_array import create_dataloader
from .data.datamanager import get_datamanager

# train functions
from .model.train import train, test

from .model.get_model import save_model

# helpers
from .helpers.measures import accuracy, f1, auroc
from .helpers.get_pool_predictions import get_pool_predictions
from .helpers.sampler import DDU_sampler

# ddu stuff

# Importing GMM utilities
from .DDU.utils.gmm_utils import get_embeddings, gmm_evaluate, gmm_fit

# Import network architectures
from .DDU.net.resnet import resnet18


def create_log_path():
    current_time = datetime.now().strftime("%H-%M-%S")
    log_file_name = "Experiment-from-" + str(current_time) + ".csv"

    log_dir = os.path.join(".", "log_dir")

    if os.path.exists(log_dir) == False:
        os.mkdir(os.path.join(".", "log_dir"))

    log_path = os.path.join(log_dir, log_file_name)
    return log_path


def save_logs(data_manager, log_path):
    log_df = data_manager.get_logs()
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


def experiment(param_dict, oracle, data_manager, writer, dataset, net, checkpoint=None):
    """experiment [Experiment function which performs the entire acitve learning process based on the predefined config]

    [extended_summary]

    Args:
        param_dict ([dict]): [experiment config from json file]
        data_manager ([class]): [Data manager which handels the management of both the dataset and the OOD data. Logs, Samples, performs oracle steps etc.]
        net ([nn.module]): [Pytorch Neural Network for experiment]

    Returns:
        [None]: [Log Attribute in Datamanage writes log to dist]
    """

    oracle_stepsize = param_dict["oracle_stepsize"]
    oracle_steps = param_dict["oracle_steps"]
    epochs = param_dict["epochs"]
    batch_size = param_dict["batch_size"]
    weight_decay = 5e-4
    metric = param_dict["metric"]
    validation_split = param_dict.get("validation_split", None)
    validation_source = param_dict.get("validation_source", None)
    lr_sheduler = param_dict["lr_sheduler"]
    verbose = param_dict["verbose"]
    metric = "accuracy"

    sampler = DDU_sampler

    log_path = create_log_path()

    if torch.cuda.is_available():
        net.cuda()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    data_manager.reset_pool()

    if torch.cuda.is_available():
        cudnn.benchmark = True

    for i in tqdm(range(oracle_steps)):

        data_loader_tuple = create_dataloader(
            data_manager,
            batch_size=batch_size,
            validation_source=validation_source,
            validation_split=validation_split,
        )

        if validation_source is not None:
            train_loader, test_loader, pool_loader, val_loader = data_loader_tuple
        else:
            train_loader, test_loader, pool_loader = data_loader_tuple
            val_loader = None

        if torch.cuda.is_available():
            net.cuda()

        criterion = nn.CrossEntropyLoss()

        optimizer = torch.optim.Adam(net.parameters(), weight_decay=weight_decay)

        if checkpoint is not None:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        net.train()
        trained_net, avg_train_loss = train(
            net,
            train_loader,
            optimizer,
            criterion,
            device=device,
            epochs=epochs,
            verbose=verbose,
            do_validation=True,
            val_dataloader=val_loader,
            patience=10,
            lr_sheduler=lr_sheduler,
        )

        avg_test_loss = test(
            trained_net, criterion, test_loader, device=device, verbose=verbose
        )

        net.eval()
        embeddings, labels = get_embeddings(
            trained_net,
            train_loader,
            num_dim=512,
            dtype=torch.double,
            device="cuda",
            storage_device="cuda",
        )

        print("fitting gmm")
        gaussians_model, jitter_eps = gmm_fit(
            embeddings=embeddings, labels=labels, num_classes=10
        )

        if len(pool_loader) > 0:
            # unlabelled pool predictions
            pool_predictions, pool_labels_list = get_pool_predictions(
                trained_net, pool_loader, device=device, return_labels=True
            )
            # class_prob = class_probs(train_loader)
            pool_predictions = torch.from_numpy(pool_predictions).cuda()
            print("finished pool prediction")
            logits, labels = gmm_evaluate(
                trained_net,
                gaussians_model,
                pool_loader,
                device=device,
                num_classes=10,
                storage_device="cuda",
            )
            # logits, labels = (
            #     logits.detach().to("cpu").numpy(),
            #     labels.detach().to("cpu").numpy(),
            # )
            print("finished gmm evaluation")
            # samples from unlabelled pool predictions
            sampler(
                dataset_manager=data_manager,
                number_samples=oracle_stepsize,
                net=trained_net,
                predictions=pool_predictions,
                gmm_logits=logits,
                class_probs=pool_predictions,
            )

            print("sampled images")
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
            print(dict_to_add)

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

        data_manager.status_manager.to_csv(f"statusmanager_oraclestep_{i}.csv")

        save_model(
            net,
            optimizer,
            param_dict,
            data_manager,
            f"model_in_step_{i}",
            in_dist=["Cifar10"],
            ood_data=["FashionMNIST", "MNIST"],
        )
        save_logs(data_manager, log_path)
    return net, optimizer


def start_experiment(config_path, log):
    """start_experiment [function which starts all experiments in the config json - main function of this module]

    [extended_summary]

    Args:
        config_path ([String]): [path to the json config]
        log ([String]): [path to log folder]
    """

    writer = SummaryWriter()

    with open(config_path, mode="r", encoding="utf-8") as config_f:
        config = json.load(config_f)

    in_dist_data = config["in_dist_data"]
    ood_data = config["ood_data"]

    data_manager = ""
    for dataset in in_dist_data:
        for exp in config["experiment-list"]:
            if exp["verbose"] > 1:
                print("Experiment Config :")
                for variable in exp:
                    if exp[variable] is not None:
                        print(f"{variable} : ", exp[variable])

            del data_manager
            gc.collect()
            data_manager = get_datamanager(indistribution=in_dist_data, ood=ood_data)

            metric = exp["metric"]

            for oracle in exp["oracles"]:

                net = resnet18(spectral_normalization=True, mod=True)

                data_manager.create_merged_data(
                    test_size=exp["test_size"],
                    pool_size=exp["pool_size"],
                    labelled_size=exp["labelled_size"],
                    OOD_ratio=exp["OOD_ratio"],
                )
                if exp["load_model"]:
                    checkpoint = torch.load(os.path.join(exp["path_model"]))
                    net.load_state_dict(checkpoint["model_state_dict"])
                    data_manager.status_manager = pd.read_csv(
                        os.path.join(exp["path_status_manager"])
                    )
                    data_manager.load_logs(exp["path_to_logs"])
                else:
                    checkpoint = None

                trained_net, optimizer = experiment(
                    param_dict=exp,
                    oracle=oracle,
                    data_manager=data_manager,
                    writer=writer,
                    dataset=dataset,
                    net=net,
                    checkpoint=checkpoint,
                )
                model_dir = os.path.join(".", "saved_models")
                if os.path.exists(model_dir) == False:
                    os.mkdir(os.path.join(".", "saved_models"))

                current_time = datetime.now().strftime("%H-%M-%S")
                if exp.get("do_save_model", False):
                    save_model(
                        trained_net,
                        optimizer,
                        exp,
                        data_manager,
                        model_dir,
                        in_dist_data,
                        ood_data,
                        desc_str="Experiment-from-" + str(current_time),
                    )

            writer.close()
    print(
        """
    **********************************************


                  EXPERIMENT DONE

    **********************************************
    """
    )
