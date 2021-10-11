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
import numpy as np
import json


# data imports
from .data.datahandler_for_array import create_dataloader
from .data.datamanager import get_datamanager

# train functions
from .model.train import train, test, get_density_vals

from .model.get_model import get_model, save_model
from .model.model_files.Angular_Penalty_Softmax_Losses import AngularPenaltySMLoss

# helpers
from .helpers.measures import accuracy, f1, auroc
from .helpers.get_pool_predictions import get_pool_predictions

from .helpers.get_tsne_plot import get_tsne_plot
from .helpers.get_density_plot import density_plot


do_tsne = False


def experiment(param_dict, oracle, data_manager, writer, dataset, net):
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
    weight_decay = param_dict["weight_decay"]
    metric = param_dict["metric"]
    lr = param_dict["lr"]
    nesterov = param_dict["nesterov"]
    momentum = param_dict["momentum"]
    validation_split = param_dict.get("validation_split", None)
    validation_source = param_dict.get("validation_source", None)
    lr_sheduler = param_dict["lr_sheduler"]
    verbose = param_dict["verbose"]
    do_pertubed_images = param_dict["do_pertubed_images"]
    do_desity_plot = param_dict["do_desity_plot"]
    criterion = param_dict["criterion"]
    model_name = param_dict['model_name']



    if param_dict.get("bugged_and_working",None) is None:
        bugged_and_working = param_dict.get("bugged_and_working",True)
        print(f"INFO ---- flag bugged_and_working is not set. Using default value of {bugged_and_working}")
    else:  
        bugged_and_working = param_dict["bugged_and_working"]
        print(f"INFO ---- flag bugged_and_working is set to {bugged_and_working}")
    
    OoD_extra_class = param_dict.get("OoD_extra_class", False)
    if OoD_extra_class:
        extra_class_thresholding = param_dict.get("extra_class_thresholding",'soft')
        print(f'INFO --- Training OoD as extra class with {extra_class_thresholding} Thresholding')
        assert param_dict.get('similarity',None) is None, f"similarity must be None, found {param_dict.get('similarity',None)}"
        assert oracle=="extra_class_entropy", f"Only extra_class_entropy oracle is supported found {oracle}"
    else:
        extra_class_thresholding = None
    
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
        raise NotImplementedError
    elif oracle == "Gen0din":
        from .helpers.sampler import gen0din_sampler

        raise NotImplementedError
        # sampler = gen0din_sampler
    elif oracle == 'extra_class_entropy':
        from .helpers.sampler import extra_class_sampler

        sampler = extra_class_sampler(extra_class_thresholding)

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

        if criterion in ["arcface", "sphereface", "cosface"]:
            in_features = 10
            num_classes = 10
            criterion = AngularPenaltySMLoss(in_features, num_classes, criterion)
        else:
            criterion = nn.CrossEntropyLoss()

        if model_name in ["gen_odin_conv","gen_odin_res","small_gen_odin_res","small_resnet_with_spec","base_small_resnet"]:
            base_params = []
            gen_odin_params = []
            for name, param in net.named_parameters():
                if name not in [
                # "g_func.weight",
                    "h_func.bias",
                # "g_norm.weight",
                # "g_norm.bias",
                    "h_func.weights",
                    "scaling_factor",
                ]:
                    base_params.append(param)  # can't do the name tupel
                else:
                    if verbose >= 2:
                        print("added name: ", name)
                    gen_odin_params.append(param)

            
            optimizer = optim.SGD(
                [
                    {"params": base_params},
                    {"params": gen_odin_params, "weight_decay": 0.0},
                ],
                weight_decay=weight_decay,
                lr=lr,
                momentum=momentum,
                nesterov=nesterov,)
        else:
            optimizer = optim.SGD(
                net.parameters(),
                weight_decay=weight_decay,
                lr=lr,
                momentum=momentum,
                nesterov=nesterov,
            )

        # optimizer = optim.SGD(
        #     net.parameters(),
        #     weight_decay=weight_decay,
        #     momentum=momentum,
        #     lr=lr,
        #     nesterov=nesterov,
        # )

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
        if do_desity_plot:
            if model_name not in ['base_small_resnet']:
                print('\n\n\WARNING ---------------------------------------------------------------------------',\
                        f'Doing density plots while using model {model_name} is not supported\n\n')
            pert_preds, gs, hs, targets = get_density_vals(
                pool_loader, val_loader, trained_net, do_pertubed_images, bugged_and_working
            )

            density_plot(pert_preds, gs, hs, targets, writer, i)
            
        if len(pool_loader) > 0:
            if do_desity_plot:
                pool_predictions, pool_labels_list, weighting_factors = (np.concatenate(pert_preds,axis=0),
                                                                        np.concatenate(targets,axis=0), 
                                                                        np.concatenate(gs,axis=0))
            else:
                # unlabelled pool predictions
                pool_predictions, pool_labels_list, weighting_factors = get_pool_predictions(
                trained_net, pool_loader, device=device, return_labels=True, bugged_and_working=bugged_and_working
                )


            if model_name=='gram_resnet':
                from .model.model_files.gram_resnet import Detector

                dector = Detector()
                POWERS = range(1,11)
                dector.compute_minmaxs(trained_net,train_loader,POWERS=POWERS)
                pool_deviations = dector.compute_deviations(trained_net,val_loader,POWERS=POWERS)
                if validation_source is not None:
                    validation = dector.compute_deviations(trained_net,val_loader,POWERS=POWERS)
                    t95 = validation.mean(axis=0)+10**-7
                    weighting_factors = (pool_deviations/t95[np.newaxis,:]).sum(axis=1)
        
                weighting_factors = np.exp(-weighting_factors)

            if (weighting_factors is not None) and (len(weighting_factors)==0):
                weighting_factors = None
            
            if bugged_and_working:
                print(f'Weighting factors are not used as bugged_and_working flag is {bugged_and_working}')
                weighting_factors = None


            # samples from unlabelled pool predictions
            sampler(
                dataset_manager=data_manager,
                number_samples=oracle_stepsize,
                net=trained_net,
                predictions=pool_predictions,
                weights = weighting_factors
            )

        test_predictions, test_labels, _ = get_pool_predictions(
            trained_net, test_loader, device=device, return_labels=True
        )
        train_predictions, train_labels, _  = get_pool_predictions(
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

        # data_manager.add_log(log_dict=dict_to_add)
        # print(dict_to_add)

        # if oracle is not None:
        #     predictions = get_pool_predictions(trained_net, pool_loader, device=device)
        #     oracle(
        #         dataset_manager=data_manager,
        #         number_samples=oracle_stepsize,
        #         predictions=predictions,
        #     )

    return net, optimizer


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

    in_dist_data = config["in_dist_data"]
    ood_data = config["ood_data"]
    for dataset in in_dist_data:

        for exp in config["experiment-list"]:
            if exp["verbose"] > 1:
                print("Experiment Config :")
                for variable in exp:
                    if exp[variable] is not None:
                        print(f"{variable} : ", exp[variable])

            OoD_extra_class = exp.get("OoD_extra_class", False)

            data_manager = get_datamanager(indistribution=in_dist_data, ood=ood_data,
                                             OoD_extra_class=OoD_extra_class)

            metric = exp["metric"]

            for oracle in exp["oracles"]:
            
                if OoD_extra_class:
                    num_classes=11
                else:
                    num_classes=10

                net = get_model(
                    exp["model_name"],
                    similarity=exp["similarity"],
                    num_classes=num_classes,
                    include_bn=False,
                    channel_input=3,
                )

                data_manager.create_merged_data(
                    test_size=exp["test_size"],
                    pool_size=exp["pool_size"],
                    labelled_size=exp["labelled_size"],
                    OOD_ratio=exp["OOD_ratio"],
                )

                trained_net, optimizer = experiment(
                    param_dict=exp,
                    oracle=oracle,
                    data_manager=data_manager,
                    writer=writer,
                    dataset=dataset,
                    net=net,
                )

                log_df = data_manager.get_logs()

                current_time = datetime.now().strftime("%H-%M-%S")
                log_file_name = "Experiment-from-" + str(current_time)+ "-" + str(exp["similarity"])

                log_dir = os.path.join(".", "log_dir")

                model_dir = os.path.join(".", "saved_models")

                if os.path.exists(log_dir) == False:
                    os.mkdir(os.path.join(".", "log_dir"))

                if os.path.exists(model_dir) == False:
                    os.mkdir(os.path.join(".", "saved_models"))

                if exp.get("do_save_model", False):
                    save_model(
                        trained_net,
                        optimizer,
                        exp,
                        data_manager,
                        model_dir,
                        in_dist_data,
                        ood_data,
                        desc_str=log_file_name+'.csv'
                    )

                log_path = os.path.join(log_dir, log_file_name+'.csv')

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
                
                log_config_path = os.path.join(log_dir, log_file_name+'.json')
                with open(log_config_path, 'w') as f:
                    json.dump(exp, f)
            writer.close()
    print(
        """
    **********************************************


                  EXPERIMENT DONE

    **********************************************
    """
    )
