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
from .model import train


from .model.get_model import get_model

# helpers
from .helpers.accuracy import accuracy
from .helpers.get_pool_predictions import get_pool_predictions

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#from . import test


def experiment(param_dict, data_manager, net, verbose=0):
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

        trained_net, avg_train_loss = train.train(
            net, train_loader, optimizer, criterion,device=device ,epochs=epochs, verbose=verbose
        )
        
        avg_test_loss = train.test(trained_net, criterion, test_loader,device=device, verbose=verbose)
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

    with open(config_path, mode="r", encoding="utf-8") as config_f:
        config = json.load(config_f)

    datasets = config["datasets"]
    for dataset in datasets:
        
        data_manager = get_datamanager(dataset=dataset)

        for exp in config["experiment-list"]:
            try:
                net = get_model(exp["model_name"])
            except Exception as e:
                print(e)
                continue

            data_manager.create_merged_data(test_size=exp["test_size"], pool_size=exp["pool_size"], labelled_size=exp["labelled_size"], OOD_ratio=exp["OOD_ratio"])

            experiment(param_dict=exp, net=net, verbose=0, data_manager=data_manager)

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
