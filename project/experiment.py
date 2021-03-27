import json
import torch

import torch.nn as nn
import torch.optim as optim

from datetime import datetime
import os
from tqdm import tqdm

# data imports
from .data.get_dataloader import get_dataloader
from .data.get_datamanager import get_datamanager

# train functions
from .model.train import train, test
from .model.mnist_model import Net
from .model.genOdinModel import genOdinModel

# helpers
from .helpers.accuracy import accuracy
from .helpers.get_pool_predictions import get_pool_predictions

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



def experiment(param_dict, data_manager, net, verbose=0):
    oracle_stepsize = param_dict["oracle_stepsize"]
    oracle_steps = param_dict["oracle_steps"]
    epochs = param_dict["epochs"]
    batch_size = param_dict["batch_size"]
    oracle = None#param_dict["oracle"]
    weight_decay = param_dict["weight_decay"]


    
    data_manager.reset_pool()

    for i in tqdm(range(oracle_steps)):
        train_loader, test_loader, pool_loader = get_dataloader(
            data_manager, batch_size=batch_size
        )

        net = net
        if torch.cuda.is_available():
            net.cuda()
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(net.parameters(), weight_decay=weight_decay)

        trained_net, avg_train_loss = train(
            net, train_loader, optimizer, criterion,device=device ,epochs=epochs, verbose=verbose
        )

        avg_test_loss = test(trained_net, criterion, test_loader,device=device, verbose=verbose)
        test_predictions, test_labels = get_pool_predictions(
            trained_net, test_loader,device=device, return_labels=True
        )
        train_predictions, train_labels = get_pool_predictions(
            trained_net, train_loader,device=device, return_labels=True
        )

        test_accuracy = accuracy(test_labels, test_predictions)
        train_accuracy = accuracy(train_labels, train_predictions)

        dict_to_add = {
            "test_loss": avg_test_loss,
            "train_loss": avg_train_loss,
            "test_accuracy": test_accuracy,
            "train_accuracy": train_accuracy,
        }

        data_manager.add_log(log_dict=dict_to_add)
        print(dict_to_add)

        if oracle is not None:
            predictions = get_pool_predictions(trained_net, pool_loader,device=device)
            oracle(
                dataset_manager=data_manager,
                number_samples=oracle_stepsize,
                predictions=predictions,
            )

    return None


def start_experiment(config_path, log):
    # print(config_path)
    # print(log)
    log_dir = log
    config = ""

    with open(config_path, mode="r", encoding="utf-8") as config_f:
        config = json.load(config_f)
    print(config)
    data_manager = get_datamanager(dataset=config["dataset"])
    
    data_manager.create_merged_data(test_size=config["test_size"], pool_size=config["pool_size"], labelled_size=config["labelled_size"], OOD_ratio=config["OOD_ratio"])

    # TODO modulizing NN as config param
    net = Net()

    genOdinModel = genOdinModel()

    experiment(param_dict=config, net=net, verbose=0, data_manager=data_manager)

    log_df = data_manager.get_logs()

    current_time = datetime.now().strftime("%H-%M-%S")
    log_file_name = "Experiment-from-" + str(current_time) + ".csv"

    if not os.path.exists("project\log"):
        os.mkdir(os.path.join(".", "log_dir"))

    log_dir = os.path.join("project\log")
    log_path = os.path.join(log_dir, log_file_name)

    with open(log_path, mode="w", encoding="utf-8") as logfile:
        colums = log_df.columns
        for colum in colums:
            logfile.write(colum + ",")
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