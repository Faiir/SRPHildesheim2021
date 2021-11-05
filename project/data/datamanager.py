import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

import torchvision.transforms as transforms

from torchvision.datasets import MNIST, FashionMNIST, SVHN, CIFAR10
from torch.utils.data import Subset, ConcatDataset
import copy
import time
import gc

# from .tinyimagenetloader import (
#     TrainTinyImageNetDataset,
#     TestTinyImageNetDataset,
#     download_and_unzip,
# )

import os

# from ..helpers.memory_tracer import display_top
import tracemalloc

from collections import Counter
import linecache
import os
import tracemalloc


def display_top(snapshot, key_type="lineno", limit=10):
    snapshot = snapshot.filter_traces(
        (
            tracemalloc.Filter(False, "<frozen importlib._bootstrap>"),
            tracemalloc.Filter(False, "<unknown>"),
        )
    )
    top_stats = snapshot.statistics(key_type)

    print("Top %s lines" % limit)
    for index, stat in enumerate(top_stats[:limit], 1):
        frame = stat.traceback[0]
        # replace "/path/to/module/file.py" with "module/file.py"
        filename = os.sep.join(frame.filename.split(os.sep)[-2:])
        print(
            "#%s: %s:%s: %.1f KiB" % (index, filename, frame.lineno, stat.size / 1024)
        )
        line = linecache.getline(frame.filename, frame.lineno).strip()
        if line:
            print("    %s" % line)

    other = top_stats[limit:]
    if other:
        size = sum(stat.size for stat in other)
        print("%s other: %.1f KiB" % (len(other), size / 1024))
    total = sum(stat.size for stat in top_stats)
    print("Total allocated size: %.1f KiB" % (total / 1024))


debug = False


class Data_manager:
    """
    Datamanager which is the backbone of the active learning pipeline and keeps track of the images used & samples as well as logs the results.
    Inputs:
        iD_datasets : list of iD datasets
        OoD_datasets : list of OoD datasets
        OoD_ratio : ratio of OoD samples in pool
        OoD_extra_class : flag to use OoD as extra class
    """

    def __init__(
        self,
        iD_datasets,
        OoD_datasets,
        labelled_size,
        unlabelled_size,
        OoD_ratio,
        test_iD_size = None
    ):

        self.iD_datasets = iD_datasets
        self.OoD_datasets = OoD_datasets
        self.OoD_ratio = OoD_ratio
        self.OoD_extra_class = False
        self.labelled_size = labelled_size
        self.unlabelled_size = unlabelled_size


        assert len(self.iD_datasets)==1, f"Only one dataset can be in-Dist, found {self.iD_datasets}"

        list_of_datasets = self.iD_datasets+self.OoD_datasets
        self.datasets_dict = data_loader(list_of_datasets)
        
        self.iD_samples_size = 0
        for ii in self.iD_datasets:
            self.iD_samples_size += len(self.datasets_dict[ii+"_train"].targets)

        self.OoD_samples_size = 0
        for ii in self.OoD_datasets:
            self.OoD_samples_size += len(self.datasets_dict[ii+"_train"].targets)

        test_dataset_iD_size = 0
        for ii in self.iD_datasets:
            test_dataset_iD_size += len(self.datasets_dict[ii+"_test"].targets)

        if test_iD_size is None:
            self.test_iD_size = test_dataset_iD_size
        else:
            assert test_dataset_iD_size >= test_iD_size
            self.test_iD_size = test_iD_size


        print(f'INFO ----- Total iD samples for training  {self.iD_samples_size}')
        print(f'INFO ----- Total iD samples for testing  {self.test_iD_size}')
        print(f'INFO ----- Total OoD samples for training {self.OoD_samples_size}')
        

    def create_merged_data(self):
        print("Creating New Dataset")

        assert 0 <= self.OoD_ratio < 1, "Invalid OOD_ratio : {self.OoD_ratio}"
        

        iD_pool_size = int(self.unlabelled_size*self.OoD_ratio)
        OoD_pool_size = self.unlabelled_size - iD_pool_size

        assert (
            self.labelled_size + iD_pool_size <= self.iD_samples_size
        ), f"Insufficient Samples in Base Dataset: labelled_size + iD_pool_size > iD_samples_size : {self.labelled_size} + {iD_pool_size} > {self.iD_samples_size}"




        iD_dataset_name = self.iD_datasets[0]+"_train"
        iD_labels = self.datasets_dict[iD_dataset_name].targets

        if iD_pool_size > 0:
            labelled_inds, pool_iD_inds, labelled_labels, pool_iD_labels  = train_test_split(np.arange(self.iD_samples_size),
                                                                                             iD_labels,
                                                                                             train_size=self.labelled_size,
                                                                                             test_size=iD_pool_size,
                                                                                             stratify=iD_labels)

            index_list = [labelled_inds, pool_iD_inds]
            targets_list = [labelled_labels, pool_iD_labels]
            status_list = [
                np.ones_like(labelled_labels),
                np.zeros_like(pool_iD_labels),
            ]
            source_list = [
                np.ones_like(labelled_labels),
                np.ones_like(pool_iD_labels),
            ]
            
            dataset_list = [
                np.repeat(iD_dataset_name,len(labelled_inds)),
                np.repeat(iD_dataset_name,len(pool_iD_inds)),
            ]

        else:
            print("Running Experiment without Pool")

            labelled_inds, pool_iD_inds, labelled_labels, pool_iD_labels  = train_test_split(np.arange(self.iD_samples_size),
                                                                                             iD_labels,
                                                                                             train_size=self.labelled_size,
                                                                                             test_size=len(np.unique(iD_labels)),
                                                                                             stratify=iD_labels)


            index_list = [labelled_inds]
            targets_list = [labelled_labels]
            status_list = [np.ones_like(labelled_labels)]
            source_list = [np.ones_like(labelled_labels)]
            dataset_list = [np.repeat(iD_dataset_name,len(labelled_inds))]


        if OoD_pool_size > 0:
            OoD_source_list = []
            OoD_inds_list = []
            OoD_label_list = []
            for ii in self.OoD_datasets:
                targets = self.datasets_dict[ii+'_train'].targets
                length = len(targets)
                OoD_source_list.append(np.repeat(ii+'_train',length))
                OoD_inds_list.append(np.arange(length))
                OoD_label_list.append(targets)

            OoD_inds_list = np.concatenate(OoD_inds_list,axis=0)
            OoD_source_list = np.concatenate(OoD_source_list,axis=0)
            OoD_label_list = np.concatenate(OoD_label_list,axis=0)      

            OoD_inds, _, OoD_source, _ = train_test_split(OoD_inds_list,
                                                          OoD_source_list,
                                                          train_size=OoD_pool_size,
                                                          stratify=OoD_label_list)
            index_list.append(OoD_inds)
            targets_list.append(-np.ones_like(OoD_inds))
            status_list.append(np.zeros_like(OoD_inds))
            source_list.append(-np.ones_like(OoD_inds))
            dataset_list.append(OoD_source)
        else:
            pass

        pool_targets = np.concatenate(targets_list)
        pool_inds = np.concatenate(index_list)
        pool_status = np.concatenate(status_list)
        pool_source = np.concatenate(source_list)
        pool_dataset = np.concatenate(dataset_list)

        
        self.status_manager = pd.DataFrame(
            np.concatenate(
                [
                    pool_inds[...,np.newaxis],
                    pool_targets[..., np.newaxis],
                    pool_source[..., np.newaxis],
                    pool_status[..., np.newaxis],
                    pool_dataset[..., np.newaxis]
                ],
                axis=1,
            ),
            columns=['inds',"target", "source", "status","dataset_name"],
        )        

        for ii in ['inds',"target", "source", "status"]:
            self.status_manager[ii] = self.status_manager[ii].astype(int)

        train_labels = self.status_manager.loc[(self.status_manager["source"].values == 1),"target"]
        OoD_class_label = max(train_labels) + 1
        self.status_manager["target"] = np.where(self.status_manager["source"].values == 1,
                                                 self.status_manager["target"].values,
                                                 OoD_class_label)

        self.iter = 0
        
        self.config = {
            "Total_overall_examples": len(self.status_manager),
            "Total_base_examples": len(
                self.status_manager[self.status_manager["source"] > 0]
            ),
            "Total_OOD_examples": len(
                self.status_manager[self.status_manager["source"] < 0]
            ),
            "Initial_examples_labelled": len(
                self.status_manager[self.status_manager["status"] == 1]
            ),
        }

        self.log = {}

        #self.save_experiment_start(csv=save_csv)
        print("Status_manager intialised")

        return None

    def save_experiment_start(self, csv=False):
        assert (
            self.status_manager is not None
        ), "Initialise Experiment first Call create_merged_data()"

        self.experiment_setup = copy.deepcopy(self.status_manager)
        self.experiment_config = copy.deepcopy(self.config)
        print("Experiment_Setup saved")

        if csv != False:
            self.experiment_config.to_csv(f"Expermimentsetup_{time.today()}")

    def restore_experiment_start(self):
        toe = self.config["Total_overall_examples"]
        tbe = self.config["Total_base_examples"]
        toode = self.config["Total_OOD_examples"]
        iel = self.config["Initial_examples_labelled"]

        self.status_manager = self.experiment_setup
        print(
            f"Restored following config \nTotal_overall_examples: {toe} \nTotal_base_examples: {tbe} \nTotal_OOD_examples: {toode}\n Initial_examples_labelled: {iel}   "
        )

    def get_train_dataset(self):
        """get_train_data [returns the current state of the trainingspool]"""
        ## Returns all data that has been labelled so far

        assert (
            self.iter is not None
        ), "Dataset not initialized. Call create_merged_data()"

        if self.OoD_extra_class:
            labelled_mask = self.status_manager[
                self.status_manager["status"] != 0
            ].index
        else:
            labelled_mask = self.status_manager[self.status_manager["status"] > 0].index

        inds_dict = self.status_manager.iloc[labelled_mask].groupby('dataset_name')['inds'].agg(list).to_dict()

        return dataset_creator(inds_dict,self.datasets_dict)

    def get_ood_dataset(self):
        """get_ood_data [returns the current state of the out-of-distribution data in the unlabelled pool]"""
        assert (
            self.iter is not None
        ), "Dataset not initialized. Call create_merged_data()"

        labelled_ood_mask = self.status_manager[self.status_manager["status"] < 0].index
        
        inds_dict = self.status_manager.iloc[labelled_ood_mask].groupby('dataset_name')['inds'].agg(list).to_dict()

        return dataset_creator(inds_dict,self.datasets_dict)

    def get_test_dataset(self):
        """get_test_data [returns the testset]"""

        assert (
            self.iter is not None
        ), "Dataset not initialized. Call create_merged_data()"

        return self.datasets_dict[self.iD_datasets[0]+'_test']

    def get_unlabelled_pool_dataset(self):
        """get_unlabelled_pool_data [returns the state of the unlabelled pool]"""
        assert (
            self.iter is not None
        ), "Dataset not initialized. Call create_merged_data()"

        unlabelled_mask = self.status_manager[self.status_manager["status"] == 0].index

        inds_dict = self.status_manager.iloc[unlabelled_mask].groupby('dataset_name')['inds'].agg(list).to_dict()

        return dataset_creator(inds_dict,self.datasets_dict)

    def add_log(self, writer, oracle, dataset, metric, param_dict, log_dict=None):
        self.iter += 1
        #
        current_iter_log = {
            "Base_examples_labelled": len(
                self.status_manager[self.status_manager["status"] > 1]
            ),
            "OOD_examples_labelled": len(
                self.status_manager[self.status_manager["status"] < 0]
            ),
        }
        print("Sampling result", current_iter_log, self.iter)
        writer.add_scalars(
            f"{metric}/{oracle}/examples_labelled",
            current_iter_log,
            self.iter,
        )

        current_iter_log["Iteration"] = self.iter
        current_iter_log["Remaining_pool_samples"] = len(
            self.status_manager[self.status_manager["status"] == 0]
        )

        ood_ratio = str(param_dict["OOD_ratio"])
        similarity = str(param_dict["similarity"]) + "_" + str(param_dict["model_name"])

        if log_dict is not None:
            if metric == "accuracy":
                acc_dict = {}
                acc_dict["test_accuracy"] = log_dict["test_accuracy"]
                acc_dict["train_accuracy"] = log_dict["train_accuracy"]

                writer.add_scalars(f"{metric}/{oracle}/{metric}", acc_dict, self.iter)
                loss_dict = {}
                loss_dict["train_loss"] = log_dict["train_loss"]
                loss_dict["test_loss"] = log_dict["test_loss"]
                writer.add_scalars(
                    f"{similarity}/{oracle}/ood_ratio-{ood_ratio}", loss_dict, self.iter
                )
            else:
                writer.add_scalars(
                    f"{similarity}/{oracle}/ood_ratio-{ood_ratio}", log_dict, self.iter
                )
            current_iter_log.update(log_dict)

        self.log[self.iter] = current_iter_log

    def get_logs(self) -> pd.DataFrame:
        log_df = pd.DataFrame.from_dict(self.log, orient="index").set_index("Iteration")
        for key in self.config.keys():
            log_df[key] = self.config[key]
        return log_df

    def reset_pool(self):
        self.log = {}
        self.iter = 0
        self.OoD_extra_class = False
        self.status_manager.loc[self.status_manager["status"] != 1, "status"] = 0


def data_loader(datasets_list: list) -> dict:
    """
    This function takes in a list of datasets to be used in the experiments
    """
    print(f"INFO ------ List of datasets being loaded are {datasets_list}")
    datasets_dict = {}
    if "CIFAR10" in datasets_list:
        datasets_dict['CIFAR10_train'] = CIFAR10(
                                                root=r"/dataset/CHIFAR10/",
                                                train=True,
                                                download=True,
                                                transform=transforms.Compose([transforms.ToTensor(),
                                                           transforms.RandomHorizontalFlip(),
                                                           transforms.RandomCrop(32, 4)]))
        datasets_dict['CIFAR10_test'] = CIFAR10(
                                                root=r"/dataset/CHIFAR10/",
                                                train=False,
                                                download=True,
                                                transform=transforms.ToTensor(),
                                                )
        
        print("INFO ----- Dataset Loaded : CIFAR10")
        datasets_list.remove("CIFAR10")
    
    if "MNIST" in datasets_list:
        mnist_transforms = transforms.Compose([transforms.Pad(2),
                                              transforms.ToTensor(),
                                              transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
                                              transforms.RandomCrop(32, 4)])
        datasets_dict['MNIST_train'] = MNIST(root=r"/dataset/MNIST",
                                            train=True,
                                            download=True,
                                            transform=mnist_transforms)
        

        datasets_dict['MNIST_test'] = MNIST(root=r"/dataset/MNIST",
                                            train=False,
                                            download=True,
                                            transform=mnist_transforms)
        
        print("INFO ----- Dataset Loaded : MNIST")
        datasets_list.remove("MNIST")
    
    if "FashionMNIST" in datasets_list:
        fmnist_transforms = transforms.Compose([transforms.Pad(2),
                                                    transforms.ToTensor(),
                                                    transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
                                                    transforms.RandomHorizontalFlip(),
                                                    transforms.RandomCrop(32, 4)])

        datasets_dict["FashionMNIST_train"] = FashionMNIST(root="/dataset/FashionMNIST",
                                            train=True,
                                            download=True,
                                            transform=fmnist_transforms)
        
        datasets_dict["FashionMNIST_test"] = FashionMNIST(root="/dataset/FashionMNIST",
                                            train=False,
                                            download=True,
                                            transform=fmnist_transforms)
        
        print("INFO ----- Dataset Loaded : FashionMNIST")
        datasets_list.remove("FashionMNIST")

    if "SVHN" in datasets_list:
        SVHN_transforms = transforms.Compose([transforms.ToTensor(),
                            transforms.Resize(32),
                            transforms.RandomCrop(32, 4)])

        datasets_dict["SVHN_train"] = SVHN(
                                        root=r"/dataset/SVHN",
                                        split="train",
                                        download=True,
                                        transform=SVHN_transforms,
                                    )
        datasets_dict["SVHN_train"].targets = datasets_dict["SVHN_train"].labels
        datasets_dict["SVHN_test"] = SVHN(root=r"/dataset/SVHN",
                                        split="test",
                                        download=True,
                                        transform=SVHN_transforms)
        
        
        datasets_dict["SVHN_test"].targets = datasets_dict["SVHN_test"].labels
        print("INFO ----- Dataset Loaded : SVHN")
        datasets_list.remove("SVHN")


    if "CIFAR100" in datasets_list:
        datasets_dict["CIFAR100_train"] = CIFAR100(root=r"/dataset/CIFAR100",
                                                   train = True,
                                                   download = True,
                                                   transform=transforms.Compose([transforms.ToTensor(),
                                                              transforms.RandomHorizontalFlip(),
                                                              transforms.RandomCrop(32, 4)]))
        
        
        datasets_dict["CIFAR100_test"] = CIFAR100(root=r"/dataset/CIFAR100",
                                                   train = False,
                                                   download = True,
                                                   transform=transforms.ToTensor(),)
        

        print("INFO ----- Dataset Loaded : CIFAR100")
        datasets_list.remove("CIFAR100")

    assert len(datasets_list)==0, f'Not all datasets have been loaded, datasets left : {datasets_list}'
    

    return datasets_dict


def dataset_creator(indices_dict, datasets_dict):
    dataset_list = []
    for dataset_name in indices_dict:
        dataset_list.append(Subset(datasets_dict[dataset_name], indices_dict[dataset_name]))

    chained_dataset = ConcatDataset(dataset_list)

    return chained_dataset 