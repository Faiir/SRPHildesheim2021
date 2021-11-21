import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torchvision.datasets.cifar import CIFAR100

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

from collections import Counter, OrderedDict
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
        pool_size,
        OoD_ratio,
        test_iD_size=None,
        subclass={"do_subclass": False},
        grayscale=False,
    ):

        self.iD_datasets = iD_datasets
        self.OoD_datasets = OoD_datasets
        self.OoD_ratio = OoD_ratio
        self.OoD_extra_class = False
        self.labelled_size = labelled_size
        self.unlabelled_size = pool_size
        self.grayscale = grayscale
        assert (
            len(self.iD_datasets) == 1
        ), f"Only one dataset can be in-Dist, found {self.iD_datasets}"

        list_of_datasets = self.iD_datasets + self.OoD_datasets
        self.datasets_dict = data_loader(list_of_datasets, grayscale=self.grayscale)

        if subclass["do_subclass"]:
            for c, dataset in enumerate(self.iD_datasets):

                cls = set(subclass["iD_classes"])
                idx_train = [
                    i
                    for i, val in enumerate(
                        self.datasets_dict[dataset + "_train"].targets
                    )
                    if val in cls
                ]
                idx_test = [
                    i
                    for i, val in enumerate(
                        self.datasets_dict[dataset + "_test"].targets
                    )
                    if val in cls
                ]

                self.datasets_dict[dataset + "_train"].data = np.compress(
                    idx_train, self.datasets_dict[dataset + "_train"].data, axis=0
                )
                self.datasets_dict[dataset + "_train"].targets = np.compress(
                    idx_train, self.datasets_dict[dataset + "_train"].targets, axis=0
                ).tolist()

                self.datasets_dict[dataset + "_test"].data = np.compress(
                    idx_test, self.datasets_dict[dataset + "_test"].data, axis=0
                )
                self.datasets_dict[dataset + "_test"].targets = np.compress(
                    idx_test, self.datasets_dict[dataset + "_test"].targets, axis=0
                ).tolist()

            for c, dataset in enumerate(self.OoD_datasets):
                cls = set(subclass["OoD_classes"])
                idx_train = [
                    i
                    for i, val in enumerate(
                        self.datasets_dict[dataset + "_train"].targets
                    )
                    if val in cls
                ]
                idx_test = [
                    i
                    for i, val in enumerate(
                        self.datasets_dict[dataset + "_test"].targets
                    )
                    if val in cls
                ]

                self.datasets_dict[dataset + "_train"].data = np.compress(
                    idx_train, self.datasets_dict[dataset + "_train"].data, axis=0
                )
                self.datasets_dict[dataset + "_train"].targets = np.compress(
                    idx_train, self.datasets_dict[dataset + "_train"].targets, axis=0
                ).tolist()

                self.datasets_dict[dataset + "_test"].data = np.compress(
                    idx_test, self.datasets_dict[dataset + "_test"].data, axis=0
                )
                self.datasets_dict[dataset + "_test"].targets = np.compress(
                    idx_test, self.datasets_dict[dataset + "_test"].targets, axis=0
                ).tolist()

        self.iD_samples_size = 0
        for ii in self.iD_datasets:
            self.iD_samples_size += len(self.datasets_dict[ii + "_train"].targets)

        self.OoD_samples_size = 0
        for ii in self.OoD_datasets:
            self.OoD_samples_size += len(self.datasets_dict[ii + "_train"].targets)

        test_dataset_iD_size = 0
        for ii in self.iD_datasets:
            test_dataset_iD_size += len(self.datasets_dict[ii + "_test"].targets)

        if test_iD_size is None:
            self.test_iD_size = test_dataset_iD_size
        else:
            assert test_dataset_iD_size >= test_iD_size
            self.test_iD_size = test_iD_size

        print(f"INFO ----- Total iD samples for training  {self.iD_samples_size}")
        print(f"INFO ----- Total iD samples for testing  {self.test_iD_size}")
        print(f"INFO ----- Total OoD samples for training {self.OoD_samples_size}")

    def create_merged_data(self, path=None):
        print("Creating New Dataset")

        assert 0 <= self.OoD_ratio < 1, "Invalid OOD_ratio : {self.OoD_ratio}"

        iD_pool_size = int(self.unlabelled_size * self.OoD_ratio)
        OoD_pool_size = self.unlabelled_size - iD_pool_size

        assert (
            self.labelled_size + iD_pool_size <= self.iD_samples_size
        ), f"Insufficient Samples in Base Dataset: labelled_size + iD_pool_size > iD_samples_size : {self.labelled_size} + {iD_pool_size} > {self.iD_samples_size}"

        iD_dataset_name = self.iD_datasets[0] + "_train"
        iD_labels = self.datasets_dict[iD_dataset_name].targets

        if iD_pool_size > 0:
            (
                labelled_inds,
                pool_iD_inds,
                labelled_labels,
                pool_iD_labels,
            ) = train_test_split(
                np.arange(self.iD_samples_size),
                iD_labels,
                train_size=self.labelled_size,
                test_size=iD_pool_size,
                stratify=iD_labels,
            )

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
                np.repeat(iD_dataset_name, len(labelled_inds)),
                np.repeat(iD_dataset_name, len(pool_iD_inds)),
            ]

        else:
            print("Running Experiment without Pool")

            (
                labelled_inds,
                pool_iD_inds,
                labelled_labels,
                pool_iD_labels,
            ) = train_test_split(
                np.arange(self.iD_samples_size),
                iD_labels,
                train_size=self.labelled_size,
                test_size=len(np.unique(iD_labels)),
                stratify=iD_labels,
            )

            index_list = [labelled_inds]
            targets_list = [labelled_labels]
            status_list = [np.ones_like(labelled_labels)]
            source_list = [np.ones_like(labelled_labels)]
            dataset_list = [np.repeat(iD_dataset_name, len(labelled_inds))]

        if OoD_pool_size > 0:
            OoD_source_list = []
            OoD_inds_list = []
            OoD_label_list = []
            for ii in self.OoD_datasets:
                targets = self.datasets_dict[ii + "_train"].targets
                length = len(targets)
                OoD_source_list.append(np.repeat(ii + "_train", length))
                OoD_inds_list.append(np.arange(length))
                OoD_label_list.append(targets)

            OoD_inds_list = np.concatenate(OoD_inds_list, axis=0)
            OoD_source_list = np.concatenate(OoD_source_list, axis=0)
            OoD_label_list = np.concatenate(OoD_label_list, axis=0)

            OoD_inds, _, OoD_source, _ = train_test_split(
                OoD_inds_list,
                OoD_source_list,
                train_size=OoD_pool_size,
                stratify=OoD_label_list,
            )
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
                    pool_inds[..., np.newaxis],
                    pool_targets[..., np.newaxis],
                    pool_source[..., np.newaxis],
                    pool_status[..., np.newaxis],
                    pool_dataset[..., np.newaxis],
                ],
                axis=1,
            ),
            columns=["inds", "target", "source", "status", "dataset_name"],
        )
        # inds -> Dataset indices for chained dataset
        # target -> num classes (iD), num classes+1 (OoD)
        # source -> Oracle Step iteration -> negative == OoD sampled
        # status -> 0 = pool , 1 = starting dataset,
        # dataset -> chained dataset source

        for ii in ["inds", "target", "source", "status"]:
            self.status_manager[ii] = self.status_manager[ii].astype(int)

        train_labels = self.status_manager.loc[
            (self.status_manager["source"].values == 1), "target"
        ]
        OoD_class_label = max(train_labels) + 1
        self.status_manager["target"] = np.where(
            self.status_manager["source"].values == 1,
            self.status_manager["target"].values,
            OoD_class_label,
        )

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

        self.status_manager.sort_values(["dataset_name"], inplace=True)
        self.status_manager.reset_index(drop=True, inplace=True)

        if path is not None:
            self.status_manager.to_csv(os.path.join(path, "intial_statusmanager.csv"))
        # self.save_experiment_start(csv=save_csv)
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
            self.experiment_setup.to_csv(f"Expermimentsetup_{time.today()}")

    def restore_experiment_start(self):
        toe = self.config["Total_overall_examples"]
        tbe = self.config["Total_base_examples"]
        toode = self.config["Total_OOD_examples"]
        iel = self.config["Initial_examples_labelled"]

        self.status_manager = self.experiment_setup
        print(
            f"Restored following config \nTotal_overall_examples: {toe} \nTotal_base_examples: {tbe} \nTotal_OOD_examples: {toode}\n Initial_examples_labelled: {iel}   "
        )

    def get_pool_source_labels(self):
        """
        Returns an binary array of source labels for pool datasamples.
        0: OoD
        1: iD  
        """
        unlabelled_mask = self.status_manager[self.status_manager["status"] == 0].index
        return np.array((self.status_manager.source[unlabelled_mask].values+1)/2,dtype=np.bool)


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

        inds_df = (
            self.status_manager.iloc[labelled_mask]
            .groupby("dataset_name", sort=False)["inds"]
            .agg(list)
        )
        inds_dict = OrderedDict()
        for ii in inds_df.index:
            inds_dict[ii] = inds_df[ii]

        return dataset_creator(inds_dict, self.datasets_dict)

    def get_ood_dataset(self):
        """get_ood_data [returns the current state of the out-of-distribution data in the unlabelled pool]"""
        assert (
            self.iter is not None
        ), "Dataset not initialized. Call create_merged_data()"

        labelled_ood_mask = self.status_manager[self.status_manager["status"] < 0].index

        inds_df = (
            self.status_manager.iloc[labelled_ood_mask]
            .groupby("dataset_name", sort=False)["inds"]
            .agg(list)
        )
        inds_dict = OrderedDict()
        for ii in inds_df.index:
            inds_dict[ii] = inds_df[ii]

        return dataset_creator(inds_dict, self.datasets_dict)

    def get_test_dataset(self):
        """get_test_data [returns the testset]"""

        assert (
            self.iter is not None
        ), "Dataset not initialized. Call create_merged_data()"

        return self.datasets_dict[self.iD_datasets[0] + "_test"]

    def get_unlabelled_pool_dataset(self):
        """get_unlabelled_pool_data [returns the state of the unlabelled pool]"""
        assert (
            self.iter is not None
        ), "Dataset not initialized. Call create_merged_data()"

        unlabelled_mask = self.status_manager[self.status_manager["status"] == 0].index

        inds_df = (
            self.status_manager.iloc[unlabelled_mask]
            .groupby("dataset_name", sort=False)["inds"]
            .agg(list)
        )
        inds_dict = OrderedDict()
        for ii in inds_df.index:
            inds_dict[ii] = inds_df[ii]

        return dataset_creator(inds_dict, self.datasets_dict)

    def get_unlabelled_iD_pool_dataset(self):
        """get_unlabelled_pool_data [returns the state of the unlabelled pool]"""
        assert (
            self.iter is not None
        ), "Dataset not initialized. Call create_merged_data()"

        unlabelled_mask = self.status_manager[(self.status_manager["status"] == 0) & (self.status_manager["source"] == 1)].index

        inds_df = (
            self.status_manager.iloc[unlabelled_mask]
            .groupby("dataset_name", sort=False)["inds"]
            .agg(list)
        )
        inds_dict = OrderedDict()
        for ii in inds_df.index:
            inds_dict[ii] = inds_df[ii]

        return dataset_creator(inds_dict, self.datasets_dict)

    def add_log(
        self, writer, oracle, dataset, metric, ood_ratio, exp_name, log_dict=None
    ):
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

        ood_ratio = str(ood_ratio)
        # similarity = str(param_dict["similarity"]) + "_" + str(param_dict["model_name"])

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
                    f"{exp_name}/{oracle}/ood_ratio-{ood_ratio}", loss_dict, self.iter
                )

                f1_scalar = np.array(log_dict["f1"])
                writer.add_scalar(
                    f"{exp_name}/{oracle}/ood_ratio-{ood_ratio}", f1_scalar, self.iter
                )
            else:
                writer.add_scalars(
                    f"{exp_name}/{oracle}/ood_ratio-{ood_ratio}", log_dict, self.iter
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


def tmp_func(x):
    return x.repeat(3, 1, 1)


def data_loader(datasets_list: list, grayscale=False) -> dict:
    """
    This function takes in a list of datasets to be used in the experiments
    """
    print(f"INFO ------ List of datasets being loaded are {datasets_list}")

    datasets_dict = {}
    if "CIFAR10" in datasets_list:
        if not grayscale:
            cifar_train_transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomCrop(32, 4),
                ]
            )
            cifar_test_transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                ]
            )
        else:
            cifar_train_transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Grayscale(3),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomCrop(32, 4),
                ]
            )
            cifar_test_transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Grayscale(3),
                ]
            )
        datasets_dict["CIFAR10_train"] = CIFAR10(
            root=r"/dataset/CHIFAR10/",
            train=True,
            download=True,
            transform=cifar_train_transform,
        )
        datasets_dict["CIFAR10_test"] = CIFAR10(
            root=r"/dataset/CHIFAR10/",
            train=False,
            download=True,
            transform=cifar_test_transform,
        )

        print("INFO ----- Dataset Loaded : CIFAR10")
        datasets_list.remove("CIFAR10")

    if "MNIST" in datasets_list:
        mnist_transforms = transforms.Compose(
            [
                transforms.Pad(2),
                transforms.ToTensor(),
                transforms.Lambda(tmp_func),
                transforms.RandomCrop(32, 4),
            ]
        )
        datasets_dict["MNIST_train"] = MNIST(
            root=r"/dataset/MNIST",
            train=True,
            download=True,
            transform=mnist_transforms,
        )

        datasets_dict["MNIST_test"] = MNIST(
            root=r"/dataset/MNIST",
            train=False,
            download=True,
            transform=mnist_transforms,
        )

        print("INFO ----- Dataset Loaded : MNIST")
        datasets_list.remove("MNIST")

    if "FashionMNIST" in datasets_list:
        fmnist_transforms = transforms.Compose(
            [
                transforms.Pad(2),
                transforms.ToTensor(),
                transforms.Lambda(tmp_func),
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, 4),
            ]
        )

        datasets_dict["FashionMNIST_train"] = FashionMNIST(
            root="/dataset/FashionMNIST",
            train=True,
            download=True,
            transform=fmnist_transforms,
        )

        datasets_dict["FashionMNIST_test"] = FashionMNIST(
            root="/dataset/FashionMNIST",
            train=False,
            download=True,
            transform=fmnist_transforms,
        )

        print("INFO ----- Dataset Loaded : FashionMNIST")
        datasets_list.remove("FashionMNIST")

    if "SVHN" in datasets_list:
        SVHN_transforms = transforms.Compose(
            [transforms.ToTensor(), transforms.Resize(32), transforms.RandomCrop(32, 4)]
        )

        datasets_dict["SVHN_train"] = SVHN(
            root=r"/dataset/SVHN",
            split="train",
            download=True,
            transform=SVHN_transforms,
        )
        datasets_dict["SVHN_train"].targets = datasets_dict["SVHN_train"].labels
        datasets_dict["SVHN_test"] = SVHN(
            root=r"/dataset/SVHN",
            split="test",
            download=True,
            transform=SVHN_transforms,
        )

        datasets_dict["SVHN_test"].targets = datasets_dict["SVHN_test"].labels
        print("INFO ----- Dataset Loaded : SVHN")
        datasets_list.remove("SVHN")

    if "CIFAR100" in datasets_list:
        datasets_dict["CIFAR100_train"] = CIFAR100(
            root=r"/dataset/CIFAR100",
            train=True,
            download=True,
            transform=transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomCrop(32, 4),
                ]
            ),
        )

        datasets_dict["CIFAR100_test"] = CIFAR100(
            root=r"/dataset/CIFAR100",
            train=False,
            download=True,
            transform=transforms.ToTensor(),
        )

        print("INFO ----- Dataset Loaded : CIFAR100")
        datasets_list.remove("CIFAR100")

    if "CIFAR10_ood" in datasets_list:

        cifar_train_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, 4),
            ]
        )
        cifar_test_transform = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )

        datasets_dict["CIFAR10_ood_train"] = CIFAR10(
            root=r"/dataset/CHIFAR10/",
            train=True,
            download=True,
            transform=cifar_train_transform,
        )
        datasets_dict["CIFAR10_ood_test"] = CIFAR10(
            root=r"/dataset/CHIFAR10/",
            train=False,
            download=True,
            transform=cifar_test_transform,
        )

        print("INFO ----- Dataset Loaded : CIFAR10_ood")
        datasets_list.remove("CIFAR10_ood")

    if "CIFAR100_ood" in datasets_list:
        datasets_dict["CIFAR100_ood_train"] = CIFAR100(
            root=r"/dataset/CIFAR100",
            train=True,
            download=True,
            transform=transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomCrop(32, 4),
                ]
            ),
        )

        datasets_dict["CIFAR100_ood_test"] = CIFAR100(
            root=r"/dataset/CIFAR100",
            train=False,
            download=True,
            transform=transforms.ToTensor(),
        )

        print("INFO ----- Dataset Loaded : CIFAR100_ood")
        datasets_list.remove("CIFAR100_ood")

    assert (
        len(datasets_list) == 0
    ), f"Not all datasets have been loaded, datasets left : {datasets_list}"

    return datasets_dict


def dataset_creator(indices_dict, datasets_dict):
    dataset_list = []
    for dataset_name in indices_dict:
        dataset_list.append(
            Subset(datasets_dict[dataset_name], indices_dict[dataset_name])
        )

    chained_dataset = ConcatDataset(dataset_list)

    return chained_dataset


# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split

# import torchvision.transforms as transforms

# from torchvision.datasets import MNIST, FashionMNIST, SVHN, CIFAR10
# import copy
# import time
# import gc

# # from .tinyimagenetloader import (
# #     TrainTinyImageNetDataset,
# #     TestTinyImageNetDataset,
# #     download_and_unzip,
# # )

# import os

# # from ..helpers.memory_tracer import display_top
# import tracemalloc

# from collections import Counter
# import linecache
# import os
# import tracemalloc


# def display_top(snapshot, key_type="lineno", limit=10):
#     snapshot = snapshot.filter_traces(
#         (
#             tracemalloc.Filter(False, "<frozen importlib._bootstrap>"),
#             tracemalloc.Filter(False, "<unknown>"),
#         )
#     )
#     top_stats = snapshot.statistics(key_type)

#     print("Top %s lines" % limit)
#     for index, stat in enumerate(top_stats[:limit], 1):
#         frame = stat.traceback[0]
#         # replace "/path/to/module/file.py" with "module/file.py"
#         filename = os.sep.join(frame.filename.split(os.sep)[-2:])
#         print(
#             "#%s: %s:%s: %.1f KiB" % (index, filename, frame.lineno, stat.size / 1024)
#         )
#         line = linecache.getline(frame.filename, frame.lineno).strip()
#         if line:
#             print("    %s" % line)

#     other = top_stats[limit:]
#     if other:
#         size = sum(stat.size for stat in other)
#         print("%s other: %.1f KiB" % (len(other), size / 1024))
#     total = sum(stat.size for stat in top_stats)
#     print("Total allocated size: %.1f KiB" % (total / 1024))


# debug = False


# class Data_manager:
#     """[Datamanager which is the backbone of the active learning pipeline]

#     [Datamanager which is the backbone of the active learning pipeline and keeps track of the images used & samples as well as logs the results]

#     """

#     ## DataManager would either get the extact data (array/tensors) or it'll have a df of filenames

#     def __init__(
#         self,
#         base_data,
#         base_data_test,
#         base_labels,
#         base_labels_test,
#         OOD_data,
#         OOD_labels,
#         OoD_extra_class,
#     ):
#         self.base_data = base_data.copy()
#         self.base_data_test = base_data_test.copy()
#         self.base_labels_test = base_labels_test.copy()
#         self.base_labels = base_labels.copy()
#         self.OOD_data = OOD_data.copy()
#         self.OOD_labels = OOD_labels.copy()
#         self.log = {}
#         self.OoD_extra_class = OoD_extra_class
#         self.iter = None
#         self.config = {}

#         print("Base-data shape: ", self.base_data.shape)
#         print("OOD_data shape: ", self.OOD_data.shape)

#     def create_merged_data(
#         self, test_size, pool_size, labelled_size, OOD_ratio, save_csv=False
#     ):
#         """create_merged_data [Creates the Active Learning pools]

#         [Creates the Active Learning pools based on the gived sizes]

#         Args:
#             test_size ([int]): [test set]
#             pool_size ([int]): [amount of unlabeled samples in the pool]
#             labelled_size ([int]): [inital size of training data]
#             OOD_ratio ([float]): [ratio of OOD images form the OOD datasets]
#             save_csv (bool, optional): [saves the state of the manager in a csv]. Defaults to False.

#         Returns:
#             [type]: [description]
#         """

#         print("Creating New Dataset")
#         if debug:
#             snapshot = tracemalloc.take_snapshot()
#             display_top(snapshot)

#         assert 0 <= OOD_ratio < 1, "Invalid OOD_ratio : {OOD_ratio}"

#         base_pool_size = pool_size

#         assert (
#             test_size + base_pool_size + labelled_size <= self.base_data.shape[0]
#         ), f"Insufficient Samples in Base Dataset: test_size + labelled_size > base_data_size : {test_size} + {labelled_size} > {self.base_data.shape[0]}"

#         # train_data, self.test_data, train_labels, self.test_labels = train_test_split(
#         #     self.base_data,
#         #     self.base_labels,
#         #     test_size=test_size,
#         #     stratify=self.base_labels,
#         # )

#         self.test_data = self.base_data_test
#         self.test_labels = self.base_labels_test
#         train_data = self.base_data
#         train_labels = self.base_labels

#         if base_pool_size > 0:

#             (
#                 labelled_data,
#                 unlabelled_data,
#                 labelled_labels,
#                 unlabelled_labels,
#             ) = train_test_split(
#                 train_data,
#                 train_labels,
#                 train_size=labelled_size,
#                 test_size=base_pool_size,
#                 stratify=train_labels,
#             )

#             data_list = [labelled_data, unlabelled_data]
#             label_list = [labelled_labels, unlabelled_labels]
#             status_list = [
#                 np.ones_like(labelled_labels),
#                 np.zeros_like(unlabelled_labels),
#             ]
#             source_list = [
#                 np.ones_like(labelled_labels),
#                 np.ones_like(unlabelled_labels),
#             ]
#             if debug:
#                 snapshot = tracemalloc.take_snapshot()
#                 display_top(snapshot)
#         else:
#             print("Running Experiment without Pool")
#             if labelled_size <= len(train_data) - len(np.unique(train_labels)):
#                 (
#                     labelled_data,
#                     unlabelled_data,
#                     labelled_labels,
#                     unlabelled_labels,
#                 ) = train_test_split(
#                     train_data,
#                     train_labels,
#                     train_size=labelled_size,
#                     test_size=len(np.unique(train_labels)),
#                     stratify=train_labels,
#                 )
#                 if debug:
#                     snapshot = tracemalloc.take_snapshot()
#                     display_top(snapshot)
#             else:
#                 labelled_data = train_data[:labelled_size]
#                 labelled_labels = train_labels[:labelled_size]

#             data_list = [labelled_data]
#             label_list = [labelled_labels]
#             status_list = [np.ones_like(labelled_labels)]
#             source_list = [np.ones_like(labelled_labels)]

#         if OOD_ratio > 0:
#             OOD_size = int(pool_size * (1 / ((1 / OOD_ratio) - 1)))
#             assert OOD_size < len(
#                 self.OOD_data
#             ), f"Insufficient Samples in OOD Dataset : OOD_size > OOD_Dataset : {OOD_size} > {len(self.OOD_data)}"

#             OOD_data, _, OOD_labels, _ = train_test_split(
#                 self.OOD_data,
#                 self.OOD_labels,
#                 train_size=OOD_size,
#                 stratify=self.OOD_labels,
#             )
#             data_list.append(OOD_data)
#             label_list.append(-np.ones_like(OOD_labels))

#             status_list.append(np.zeros_like(OOD_labels))
#             source_list.append(-np.ones_like(OOD_labels))
#         else:
#             pass

#         self.pool_data = np.concatenate(data_list)
#         pool_labels = np.concatenate(label_list)
#         pool_status = np.concatenate(status_list)
#         pool_source = np.concatenate(source_list)

#         self.status_manager = pd.DataFrame(
#             np.concatenate(
#                 [
#                     pool_labels[..., np.newaxis],
#                     pool_source[..., np.newaxis],
#                     pool_status[..., np.newaxis],
#                 ],
#                 axis=1,
#             ),
#             columns=["target", "source", "status"],
#         )

#         if self.OoD_extra_class:
#             self.status_manager["original_targets"] = self.status_manager[
#                 "target"
#             ].values
#             OoD_class_label = max(train_labels) + 1
#             self.status_manager["target"] = np.where(
#                 self.status_manager["source"].values == 1,
#                 self.status_manager["original_targets"].values,
#                 OoD_class_label,
#             )
#             print(f"Setting OoD Targets as {OoD_class_label} class")

#         self.iter = 0
#         if debug:
#             snapshot = tracemalloc.take_snapshot()
#             display_top(snapshot)
#         self.config = {
#             "Total_overall_examples": len(self.status_manager),
#             "Total_base_examples": len(
#                 self.status_manager[self.status_manager["source"] > 0]
#             ),
#             "Total_OOD_examples": len(
#                 self.status_manager[self.status_manager["source"] < 0]
#             ),
#             "Initial_examples_labelled": len(
#                 self.status_manager[self.status_manager["status"] == 1]
#             ),
#         }

#         self.log = {}

#         self.save_experiment_start(csv=save_csv)
#         print("Status_manager intialised")

#         ## need these for multiple experiemnts, setting them to None breaks that functionality
#         # self.OOD_data = None
#         # self.OOD_labels = None
#         # self.base_data = None
#         # self.base_labels = None
#         if debug:
#             snapshot = tracemalloc.take_snapshot()
#             display_top(snapshot)
#         return None

#     def save_experiment_start(self, csv=False):
#         assert (
#             self.status_manager is not None
#         ), "Initialise Experiment first Call create_merged_data()"

#         self.experiment_setup = copy.deepcopy(self.status_manager)
#         self.experiment_config = copy.deepcopy(self.config)
#         print("Experiment_Setup saved")

#         if csv != False:
#             self.experiment_config.to_csv(f"Expermimentsetup_{time.today()}")

#     def restore_experiment_start(self):
#         toe = self.config["Total_overall_examples"]
#         tbe = self.config["Total_base_examples"]
#         toode = self.config["Total_OOD_examples"]
#         iel = self.config["Initial_examples_labelled"]

#         self.status_manager = self.experiment_setup
#         print(
#             f"Restored following config \nTotal_overall_examples: {toe} \nTotal_base_examples: {tbe} \nTotal_OOD_examples: {toode}\n Initial_examples_labelled: {iel}   "
#         )

#     def get_train_data(self):
#         """get_train_data [returns the current state of the trainingspool]"""
#         ## Returns all data that has been labelled so far

#         assert (
#             self.iter is not None
#         ), "Dataset not initialized. Call create_merged_data()"

#         if self.OoD_extra_class:
#             labelled_mask = self.status_manager[
#                 self.status_manager["status"] != 0
#             ].index
#         else:
#             labelled_mask = self.status_manager[self.status_manager["status"] > 0].index

#         train_data = self.pool_data[labelled_mask]
#         train_labels = self.status_manager.iloc[labelled_mask]["target"].values

#         return train_data, train_labels

#     def get_ood_data(self):
#         """get_ood_data [returns the current state of the out-of-distribution data in the unlabelled pool]"""
#         assert (
#             self.iter is not None
#         ), "Dataset not initialized. Call create_merged_data()"

#         labelled_ood_mask = self.status_manager[self.status_manager["status"] < 0].index
#         ood_train_data = self.pool_data[labelled_ood_mask]
#         ood_train_labels = self.status_manager.iloc[labelled_ood_mask]["target"].values

#         return ood_train_data, ood_train_labels

#     def get_test_data(self):
#         """get_test_data [returns the testset]"""

#         assert (
#             self.iter is not None
#         ), "Dataset not initialized. Call create_merged_data()"

#         return self.test_data, self.test_labels

#     def get_unlabelled_pool_data(self):
#         """get_unlabelled_pool_data [returns the state of the unlabelled pool]"""
#         assert (
#             self.iter is not None
#         ), "Dataset not initialized. Call create_merged_data()"

#         unlabelled_mask = self.status_manager[self.status_manager["status"] == 0].index

#         return (
#             self.pool_data[unlabelled_mask],
#             self.status_manager.iloc[unlabelled_mask]["target"].values,
#         )

#     def add_log(self, writer, oracle, dataset, metric, param_dict, log_dict=None):
#         self.iter += 1
#         #
#         current_iter_log = {
#             "Base_examples_labelled": len(
#                 self.status_manager[self.status_manager["status"] > 1]
#             ),
#             "OOD_examples_labelled": len(
#                 self.status_manager[self.status_manager["status"] < 0]
#             ),
#         }
#         print("Sampling result", current_iter_log, self.iter)
#         writer.add_scalars(
#             f"{metric}/{oracle}/examples_labelled",
#             current_iter_log,
#             self.iter,
#         )

#         current_iter_log["Iteration"] = self.iter
#         current_iter_log["Remaining_pool_samples"] = len(
#             self.status_manager[self.status_manager["status"] == 0]
#         )

#         ood_ratio = str(param_dict["OOD_ratio"])
#         similarity = str(param_dict["similarity"]) + "_" + str(param_dict["model_name"])

#         if log_dict is not None:
#             if metric == "accuracy":
#                 acc_dict = {}
#                 acc_dict["test_accuracy"] = log_dict["test_accuracy"]
#                 acc_dict["train_accuracy"] = log_dict["train_accuracy"]

#                 writer.add_scalars(f"{metric}/{oracle}/{metric}", acc_dict, self.iter)
#                 loss_dict = {}
#                 loss_dict["train_loss"] = log_dict["train_loss"]
#                 loss_dict["test_loss"] = log_dict["test_loss"]
#                 writer.add_scalars(
#                     f"{similarity}/{oracle}/ood_ratio-{ood_ratio}", loss_dict, self.iter
#                 )
#             else:
#                 writer.add_scalars(
#                     f"{similarity}/{oracle}/ood_ratio-{ood_ratio}", log_dict, self.iter
#                 )
#             current_iter_log.update(log_dict)

#         self.log[self.iter] = current_iter_log

#     def get_logs(self) -> pd.DataFrame:
#         log_df = pd.DataFrame.from_dict(self.log, orient="index").set_index("Iteration")
#         for key in self.config.keys():
#             log_df[key] = self.config[key]
#         return log_df

#     def reset_pool(self):
#         self.log = {}
#         self.iter = 0
#         self.status_manager.loc[self.status_manager["status"] != 1, "status"] = 0


# def get_datamanager(
#     indistribution=["Cifar10"],
#     ood=["MNIST", "Fashion_MNIST", "SVHN"],
#     OoD_extra_class=False,
# ):
#     """get_datamanager [Creates a datamanager instance with the In-/Out-of-Distribution Data]

#     [List based processing of Datasets. Images are resized / croped on 32x32]

#     Args:
#         indistribution (list, optional): [description]. Defaults to ["Cifar10"].
#         ood (list, optional): [description]. Defaults to ["MNIST", "Fashion_MNIST", "SVHN"].
#         OoD_extra_calss (bool) : [flag for using OoD as extra class in training]. Defaults to False

#     Returns:
#         [datamager]: [Experiment datamanager for for logging and the active learning cycle]
#     """

#     # TODO ADD Target transform?
#     base_data = np.empty(shape=(1, 3, 32, 32))
#     base_data_test = np.empty(shape=(1, 3, 32, 32))
#     base_labels = np.empty(shape=(1,))
#     base_labels_test = np.empty(shape=(1,))

#     OOD_data = np.empty(shape=(1, 3, 32, 32))
#     OOD_labels = np.empty(shape=(1,))

#     resize = transforms.Resize(32)
#     random_crop = transforms.RandomCrop(32)

#     standard_transform = transforms.Compose(
#         [
#             transforms.ToTensor(),
#             resize,
#             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
#         ]
#     )
#     if debug:
#         tracemalloc.start()
#         snapshot = tracemalloc.take_snapshot()
#         display_top(snapshot)

#     for dataset in indistribution:
#         if dataset == "Cifar10":

#             CIFAR10_train = CIFAR10(
#                 root=r"/dataset/CHIFAR10/",
#                 train=True,
#                 download=True,
#                 transform=transforms.ToTensor(),
#             )
#             CIFAR10_test = CIFAR10(
#                 root=r"/dataset/CHIFAR10/",
#                 train=False,
#                 download=True,
#                 transform=transforms.ToTensor(),
#             )
#             # CIFAR10_train_data = CIFAR10_train.data.permute(
#             #     0, 3, 1, 2
#             # )  # .reshape(-1, 3, 32, 32)
#             # CIFAR10_test_data = CIFAR10_test.data.permute(
#             #     0, 3, 1, 2
#             # )  # .reshape(-1, 3, 32, 32)

#             CIFAR10_train_data = np.array([i.numpy() for i, _ in CIFAR10_train])
#             CIFAR10_test_data = np.array([i.numpy() for i, _ in CIFAR10_test])

#             CIFAR10_train_labels = np.array(CIFAR10_train.targets)
#             CIFAR10_test_labels = np.array(CIFAR10_test.targets)

#             base_data = np.concatenate(
#                 [base_data.copy(), CIFAR10_train_data.copy()],
#                 axis=0,
#             )

#             base_data_test = np.concatenate(
#                 [base_data_test.copy(), CIFAR10_test_data.copy()]
#             )

#             base_labels = np.concatenate(
#                 [
#                     base_labels.copy(),
#                     CIFAR10_train_labels.copy(),
#                 ],
#                 axis=0,
#             )
#             base_labels_test = np.concatenate(
#                 [
#                     base_labels_test.copy(),
#                     CIFAR10_test_labels.copy(),
#                 ]
#             )

#             del (
#                 CIFAR10_train_data,
#                 CIFAR10_test_data,
#                 CIFAR10_train_labels,
#                 CIFAR10_test_labels,
#                 CIFAR10_train,
#                 CIFAR10_test,
#             )
#             gc.collect()
#         elif dataset == "MNIST":

#             MNIST_train = MNIST(
#                 root=r"/dataset/MNIST",
#                 train=True,
#                 download=True,
#                 transform=transforms.Compose(
#                     [
#                         transforms.Pad(2),
#                         transforms.Grayscale(3),
#                         transforms.ToTensor(),
#                     ]
#                 ),
#             )
#             MNIST_test = MNIST(
#                 root=r"/dataset/MNIST",
#                 train=False,
#                 download=True,
#                 transform=transforms.Compose(
#                     [
#                         transforms.Pad(2),
#                         transforms.ToTensor(),
#                         transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
#                     ]
#                 ),
#             )
#             MNIST_train_data = np.array([i.numpy() for i, _ in MNIST_train])
#             MNIST_test_data = np.array([i.numpy() for i, _ in MNIST_test])
#             if len(dataset) > 1:
#                 MNIST_train_labels = MNIST_train.targets + np.max(base_labels)
#                 MNIST_test_labels = MNIST_test.targets + np.max(base_labels)
#             else:
#                 MNIST_train_labels = MNIST_train.targets
#                 MNIST_test_labels = MNIST_test.targets

#             base_data = np.concatenate([base_data.copy(), MNIST_train_data.copy()])

#             base_data_test = np.concatenate(
#                 [base_data_test.copy(), MNIST_test_labels.copy()]
#             )

#             base_labels = np.concatenate(
#                 [
#                     base_labels.copy(),
#                     MNIST_train_labels.copy(),
#                 ]
#             )

#             base_labels_test = np.concatenate(
#                 [
#                     base_labels_test.copy(),
#                     MNIST_test_labels.copy(),
#                 ]
#             )
#             del (
#                 MNIST_train,
#                 MNIST_test,
#                 MNIST_train_data,
#                 MNIST_test_data,
#                 MNIST_train_labels,
#                 MNIST_test_labels,
#             )
#             gc.collect()
#         elif dataset == "Fashion_MNIST":

#             Fashion_MNIST_train = FashionMNIST(
#                 root="/dataset/FashionMNIST",
#                 train=True,
#                 download=True,
#                 transform=transforms.Compose(
#                     [
#                         transforms.Pad(2),
#                         transforms.ToTensor(),
#                         transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
#                     ]
#                 ),
#             )
#             Fashion_MNIST_test = FashionMNIST(
#                 root="/dataset/FashionMNIST",
#                 train=False,
#                 download=True,
#                 transform=transforms.Compose(
#                     [
#                         transforms.Pad(2),
#                         transforms.Grayscale(3),
#                         transforms.ToTensor(),
#                     ]
#                 ),
#             )
#             Fashion_MNIST_train_data = np.array(
#                 [i.numpy() for i, _ in Fashion_MNIST_train]
#             )
#             Fashion_MNIST_test_data = np.array(
#                 [i.numpy() for i, _ in Fashion_MNIST_test]
#             )

#             if len(dataset) > 1:
#                 Fashion_MNIST_train_labels = (
#                     Fashion_MNIST_train.targets.numpy() + np.max(base_labels)
#                 )
#                 Fashion_MNIST_test_labels = Fashion_MNIST_test.targets.numpy() + np.max(
#                     base_labels
#                 )
#             else:
#                 Fashion_MNIST_train_labels = Fashion_MNIST_train.targets.numpy()
#                 Fashion_MNIST_test_labels = Fashion_MNIST_test.targets.numpy()

#             base_data = np.concatenate(
#                 [base_data.copy(), Fashion_MNIST_train_data.copy()]
#             )

#             base_data_test = np.concatenate(
#                 [base_data_test.copy(), Fashion_MNIST_test_data.copy()]
#             )

#             base_labels = np.concatenate(
#                 [
#                     base_labels.copy(),
#                     Fashion_MNIST_train_labels.copy(),
#                 ]
#             )
#             base_labels_test = np.concatenate(
#                 [
#                     base_labels_test.copy(),
#                     Fashion_MNIST_test_labels.copy(),
#                 ]
#             )
#             del (
#                 Fashion_MNIST_train,
#                 Fashion_MNIST_test,
#                 Fashion_MNIST_train_data,
#                 Fashion_MNIST_test_data,
#                 Fashion_MNIST_train_labels,
#                 Fashion_MNIST_test_labels,
#             )
#             gc.collect()
#     for ood_dataset in ood:
#         if ood_dataset == "MNIST":

#             MNIST_train = MNIST(
#                 root=r"/dataset/MNIST",
#                 train=True,
#                 download=True,
#                 transform=transforms.Compose(
#                     [
#                         transforms.Pad(2),
#                         transforms.Grayscale(3),
#                         transforms.ToTensor(),
#                     ]
#                 ),
#             )
#             MNIST_test = MNIST(
#                 root=r"/dataset/MNIST",
#                 train=False,
#                 download=True,
#                 transform=transforms.Compose(
#                     [
#                         transforms.Pad(2),
#                         transforms.Grayscale(3),
#                         transforms.ToTensor(),
#                     ]
#                 ),
#             )

#             MNIST_train_data = np.array([i.numpy() for i, _ in MNIST_train])
#             MNIST_test_data = np.array([i.numpy() for i, _ in MNIST_test])

#             MNIST_train_labels = MNIST_train.targets.numpy()
#             MNIST_test_labels = MNIST_test.targets.numpy()
#             OOD_data = np.concatenate(
#                 [OOD_data.copy(), MNIST_train_data.copy(), MNIST_test_data.copy()],
#                 axis=0,
#             )
#             OOD_labels = np.concatenate(
#                 [OOD_labels.copy(), MNIST_train_labels.copy(), MNIST_test_labels.copy()]
#             )

#             del (
#                 MNIST_train,
#                 MNIST_test,
#                 MNIST_train_data,
#                 MNIST_test_data,
#                 MNIST_train_labels,
#                 MNIST_test_labels,
#             )
#             gc.collect()
#         elif ood_dataset == "Fashion_MNIST":

#             Fashion_MNIST_train = FashionMNIST(
#                 root="/dataset/FashionMNIST",
#                 train=True,
#                 download=True,
#                 transform=transforms.Compose(
#                     [
#                         transforms.Pad(2),
#                         transforms.Grayscale(3),
#                         transforms.ToTensor(),
#                     ]
#                 ),
#             )
#             Fashion_MNIST_test = FashionMNIST(
#                 root="/dataset/FashionMNIST",
#                 train=False,
#                 download=True,
#                 transform=transforms.Compose(
#                     [
#                         transforms.Pad(2),
#                         transforms.Grayscale(3),
#                         transforms.ToTensor(),
#                     ]
#                 ),
#             )
#             Fashion_MNIST_train_data = np.array(
#                 [i.numpy() for i, _ in Fashion_MNIST_train]
#             )
#             Fashion_MNIST_test_data = np.array(
#                 [i.numpy() for i, _ in Fashion_MNIST_test]
#             )
#             Fashion_MNIST_train_labels = Fashion_MNIST_train.targets.numpy()
#             Fashion_MNIST_test_labels = Fashion_MNIST_test.targets.numpy()

#             OOD_data = np.concatenate(
#                 [
#                     OOD_data.copy(),
#                     Fashion_MNIST_train_data.copy(),
#                     Fashion_MNIST_test_data.copy(),
#                 ],
#                 axis=0,
#             )
#             OOD_labels = np.concatenate(
#                 [
#                     OOD_labels.copy(),
#                     Fashion_MNIST_train_labels.copy(),
#                     Fashion_MNIST_test_labels.copy(),
#                 ],
#             )
#             del (
#                 Fashion_MNIST_train,
#                 Fashion_MNIST_test,
#                 Fashion_MNIST_train_data,
#                 Fashion_MNIST_test_data,
#                 Fashion_MNIST_train_labels,
#                 Fashion_MNIST_test_labels,
#             )
#             gc.collect()
#         elif ood_dataset == "SVHN":
#             SVHN_train = SVHN(
#                 root=r"/dataset/SVHN",
#                 split="train",
#                 download=True,
#                 transform=standard_transform,
#             )
#             SVHN_test = SVHN(
#                 root=r"/dataset/SVHN",
#                 split="test",
#                 download=True,
#                 transform=standard_transform,
#             )
#             SVHN_train_data = SVHN_train.data
#             SVHN_test_data = SVHN_test.data
#             SVHN_train_labels = SVHN_train.labels
#             SVHN_test_labels = SVHN_test.labels

#             OOD_data = np.concatenate(
#                 [OOD_data.copy(), SVHN_train_data.copy(), SVHN_test_data.copy()], axis=0
#             )
#             OOD_labels = np.concatenate(
#                 [OOD_labels.copy(), SVHN_train_labels.copy(), SVHN_test_labels.copy()]
#             )

#             del (
#                 SVHN_train,
#                 SVHN_test,
#                 SVHN_train_data,
#                 SVHN_test_data,
#                 SVHN_train_labels,
#                 SVHN_test_labels,
#             )
#             gc.collect()
#         # elif ood_dataset == "TinyImageNet":
#         #     if not os.listdir(os.path.join(r"./dataset/tiny-imagenet-200")):
#         #         download_and_unzip()
#         #     id_dict = {}
#         #     for i, line in enumerate(
#         #         open(
#         #             os.path.join(
#         #                 r"\dataset\tiny-imagenet-200\tiny-imagenet-200\wnids.txt"
#         #             ),
#         #             "r",
#         #         )
#         #     ):
#         #         id_dict[line.replace("\n", "")] = i
#         #     normalize_imagenet = transforms.Normalize(
#         #         (122.4786, 114.2755, 101.3963), (70.4924, 68.5679, 71.8127)
#         #     )
#         #     train_t_imagenet = TrainTinyImageNetDataset(
#         #         id=id_dict, transform=transforms.Compose([normalize_imagenet, resize])
#         #     )
#         #     test_t_imagenet = TestTinyImageNetDataset(
#         #         id=id_dict, transform=transforms.Compose([normalize_imagenet, resize])
#         #     )

#     if debug:
#         snapshot = tracemalloc.take_snapshot()
#         display_top(snapshot)

#     base_data = np.delete(base_data, 0, axis=0)
#     base_data_test = np.delete(base_data_test, 0, axis=0)
#     base_labels = np.delete(base_labels, 0)
#     base_labels_test = np.delete(base_labels_test, 0)
#     OOD_data = np.delete(OOD_data, 0, axis=0)
#     OOD_labels = np.delete(OOD_labels, 0)

#     data_manager = Data_manager(
#         base_data=base_data,
#         base_labels=base_labels,
#         base_data_test=base_data_test,
#         base_labels_test=base_labels_test,
#         OOD_data=OOD_data,
#         OOD_labels=OOD_labels,
#         OoD_extra_class=OoD_extra_class,
#     )
#     # del (base_data, base_labels, OOD_data, OOD_labels)

#     gc.collect()
#     if debug:
#         snapshot = tracemalloc.take_snapshot()
#         display_top(snapshot)
#     return data_manager


# # get_datamanager(ood=["Fashion_MNIST", "MNIST"])
