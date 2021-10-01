import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

import torchvision.transforms as transforms

from torchvision.datasets import MNIST, FashionMNIST, SVHN, CIFAR10, CIFAR100
import copy
import time

# from .tinyimagenetloader import (
#     TrainTinyImageNetDataset,
#     TestTinyImageNetDataset,
#     download_and_unzip,
# )
import os


class Data_manager:

    ## DataManager would either get the extact data (array/tensors) or it'll have a df of filenames

    def __init__(self, base_data, base_labels, OOD_data, OOD_labels):
        self.base_data = base_data
        self.base_labels = base_labels
        self.OOD_data = OOD_data
        self.OOD_labels = OOD_labels
        self.log = {}
        self.iter = None
        self.config = {}

        print("Base-data shape: ", self.base_data.shape)
        print("OOD_data shape: ", self.OOD_data.shape)

    def create_merged_data(
        self, test_size, pool_size, labelled_size, OOD_ratio, save_csv=False
    ):

        print("Creating New Dataset")

        assert 0 <= OOD_ratio < 1, "Invalid OOD_ratio : {OOD_ratio}"

        base_pool_size = pool_size

        assert (
            test_size + base_pool_size + labelled_size <= self.base_data.shape[0]
        ), f"Insufficient Samples in Base Dataset: test_size + labelled_size > base_data_size : {test_size} + {labelled_size} > {self.base_data.shape[0]}"

        train_data, self.test_data, train_labels, self.test_labels = train_test_split(
            self.base_data,
            self.base_labels,
            test_size=test_size,
            stratify=self.base_labels,
        )

        if base_pool_size > 0:

            (
                labelled_data,
                unlabelled_data,
                labelled_labels,
                unlabelled_labels,
            ) = train_test_split(
                train_data,
                train_labels,
                train_size=labelled_size,
                test_size=base_pool_size,
                stratify=train_labels,
            )

            data_list = [labelled_data, unlabelled_data]
            label_list = [labelled_labels, unlabelled_labels]
            status_list = [
                np.ones_like(labelled_labels),
                np.zeros_like(unlabelled_labels),
            ]
            source_list = [
                np.ones_like(labelled_labels),
                np.ones_like(unlabelled_labels),
            ]

        else:
            print("Running Experiment without Pool")
            if labelled_size <= len(train_data) - len(np.unique(train_labels)):
                (
                    labelled_data,
                    unlabelled_data,
                    labelled_labels,
                    unlabelled_labels,
                ) = train_test_split(
                    train_data,
                    train_labels,
                    train_size=labelled_size,
                    test_size=len(np.unique(train_labels)),
                    stratify=train_labels,
                )
            else:
                labelled_data = train_data[:labelled_size]
                labelled_labels = train_labels[:labelled_size]

            data_list = [labelled_data]
            label_list = [labelled_labels]
            status_list = [np.ones_like(labelled_labels)]
            source_list = [np.ones_like(labelled_labels)]

        if OOD_ratio > 0:
            OOD_size = int(pool_size * (1 / ((1 / OOD_ratio) - 1)))
            assert OOD_size < len(
                self.OOD_data
            ), f"Insufficient Samples in OOD Dataset : OOD_size > OOD_Dataset : {OOD_size} > {len(self.OOD_data)}"

            OOD_data, _, OOD_labels, _ = train_test_split(
                self.OOD_data,
                self.OOD_labels,
                train_size=OOD_size,
                stratify=self.OOD_labels,
            )
            data_list.append(OOD_data)
            label_list.append(-np.ones_like(OOD_labels))

            status_list.append(np.zeros_like(OOD_labels))
            source_list.append(-np.ones_like(OOD_labels))
        else:
            pass

        self.pool_data = np.concatenate(data_list)
        pool_labels = np.concatenate(label_list)
        pool_status = np.concatenate(status_list)
        pool_source = np.concatenate(source_list)

        self.status_manager = pd.DataFrame(
            np.concatenate(
                [
                    pool_labels[..., np.newaxis],
                    pool_source[..., np.newaxis],
                    pool_status[..., np.newaxis],
                ],
                axis=1,
            ),
            columns=["target", "source", "status"],
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

        self.save_experiment_start(csv=save_csv)
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

    def get_train_data(self):
        ## Returns all data that has been labelled so far

        assert (
            self.iter is not None
        ), "Dataset not initialized. Call create_merged_data()"

        labelled_mask = self.status_manager[self.status_manager["status"] > 0].index
        train_data = self.pool_data[labelled_mask]
        train_labels = self.status_manager.iloc[labelled_mask]["target"].values

        return train_data, train_labels

    def get_ood_data(self):
        assert (
            self.iter is not None
        ), "Dataset not initialized. Call create_merged_data()"

        labelled_ood_mask = self.status_manager[self.status_manager["status"] < 0].index
        ood_train_data = self.pool_data[labelled_ood_mask]
        ood_train_labels = self.status_manager.iloc[labelled_ood_mask]["target"].values

        return ood_train_data, ood_train_labels

    def get_test_data(self):
        ## Returns all test data with labels

        assert (
            self.iter is not None
        ), "Dataset not initialized. Call create_merged_data()"

        return self.test_data, self.test_labels

    def get_unlabelled_pool_data(self):
        ## Returns all data in pool, None is returned instead of labels.

        assert (
            self.iter is not None
        ), "Dataset not initialized. Call create_merged_data()"

        unlabelled_mask = self.status_manager[self.status_manager["status"] == 0].index

        return (
            self.pool_data[unlabelled_mask],
            self.status_manager.iloc[unlabelled_mask]["target"].values,
        )

    def add_log(self, writer, oracle, dataset, metric, log_dict=None):
        self.iter += 1
        # "Iteration"=  self.iter,
        current_iter_log = {
            "Base_examples_labelled": len(
                self.status_manager[self.status_manager["status"] > 1]
            ),
            "OOD_examples_labelled": len(
                self.status_manager[self.status_manager["status"] < 0]
            ),
            "Remaining_pool_samples": len(
                self.status_manager[self.status_manager["status"] == 0]
            ),
        }

        writer.add_scalars(
            f"{metric}/{dataset}/{oracle}/examples_labelled",
            current_iter_log,
            self.iter,
        )

        if log_dict is not None:
            acc_dict = {}
            acc_dict["test_accuracy"] = log_dict["test_accuracy"]
            acc_dict["train_accuracy"] = log_dict["train_accuracy"]

            writer.add_scalars(
                f"{metric}/{dataset}/{oracle}/{metric}", acc_dict, self.iter
            )
            loss_dict = {}
            loss_dict["train_loss"] = log_dict["train_loss"]
            loss_dict["test_loss"] = log_dict["test_loss"]
            writer.add_scalars(
                f"{metric}/{dataset}/{oracle}/loss", loss_dict, self.iter
            )
            current_iter_log.update(log_dict)

        self.log[self.iter] = current_iter_log

    def get_logs(self):
        log_df = pd.DataFrame.from_dict(self.log, orient="index").set_index("Iteration")
        for key in self.config.keys():
            log_df[key] = self.config[key]
        return log_df

    def reset_pool(self):
        self.log = {}
        self.iter = 0
        self.status_manager.loc[self.status_manager["status"] != 1, "status"] = 0


def get_datamanager(indistribution=["Cifar10"], ood=["MNIST", "Fashion_MNIST", "SVHN"]):
    """get_datamanager [Creates a datamanager instance with the In-/Out-of-Distribution Data]

    [List based processing of Datasets. Images are resized / croped on 32x32]

    Args:
        indistribution (list, optional): [description]. Defaults to ["Cifar10"].
        ood (list, optional): [description]. Defaults to ["MNIST", "Fashion_MNIST", "SVHN"].

    Returns:
        [datamager]: [Experiment datamanager for for logging and the active learning cycle]
    """

    # TODO ADD Target transform?
    base_data = np.empty(shape=(1, 3, 32, 32))
    base_labels = np.empty(shape=(1,))

    OOD_data = np.empty(shape=(1, 3, 32, 32))
    OOD_labels = np.empty(shape=(1,))

    resize = transforms.Resize(32)
    random_crop = transforms.RandomCrop(32)

    standard_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            resize,
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    for dataset in indistribution:
        if dataset == "Cifar10":

            CIFAR10_train = CIFAR10(
                root=r"/dataset/CHIFAR10/", train=True, download=True, transform=None
            )
            CIFAR10_test = CIFAR10(
                root=r"/dataset/CHIFAR10/", train=False, download=True, transform=None
            )
            # CIFAR10_train_data = CIFAR10_train.data.permute(
            #     0, 3, 1, 2
            # )  # .reshape(-1, 3, 32, 32)
            # CIFAR10_test_data = CIFAR10_test.data.permute(
            #     0, 3, 1, 2
            # )  # .reshape(-1, 3, 32, 32)

            CIFAR10_train_data = np.swapaxes(CIFAR10_train.data, 1, 3)
            CIFAR10_test_data = np.swapaxes(CIFAR10_test.data, 1, 3)
            CIFAR10_train_labels = np.array(CIFAR10_train.targets)
            CIFAR10_test_labels = np.array(CIFAR10_test.targets)

            base_data = np.concatenate(
                [base_data, CIFAR10_train_data, CIFAR10_test_data]
            )
            base_labels = np.concatenate(
                [base_labels, CIFAR10_train_labels, CIFAR10_test_labels]
            )
        elif dataset == "MNIST":

            MNIST_train = MNIST(
                root=r"/dataset/MNIST",
                train=True,
                download=True,
                transform=transforms.Compose(
                    [
                        transforms.Pad(2),
                        transforms.ToTensor(),
                        transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
                    ]
                ),
            )
            MNIST_test = MNIST(
                root=r"/dataset/MNIST",
                train=False,
                download=True,
                transform=transforms.Compose(
                    [
                        transforms.Pad(2),
                        transforms.ToTensor(),
                        transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
                    ]
                ),
            )
            MNIST_train_data = np.array([i.numpy() for i, _ in MNIST_train])
            MNIST_test_data = np.array([i.numpy() for i, _ in MNIST_test])
            if len(dataset) > 1:
                MNIST_train_labels = MNIST_train.targets + np.max(base_labels)
                MNIST_test_labels = MNIST_test.targets + np.max(base_labels)
            else:
                MNIST_train_labels = MNIST_train.targets
                MNIST_test_labels = MNIST_test.targets

            base_data = np.concatenate([base_data, MNIST_train_data, MNIST_test_data])
            base_labels = np.concatenate(
                [base_labels, MNIST_train_labels, MNIST_test_labels]
            )

        elif dataset == "Fashion_MNIST":

            Fashion_MNIST_train = FashionMNIST(
                root="/dataset/FashionMNIST",
                train=True,
                download=True,
                transform=transforms.Compose(
                    [
                        transforms.Pad(2),
                        transforms.ToTensor(),
                        transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
                    ]
                ),
            )
            Fashion_MNIST_test = FashionMNIST(
                root="/dataset/FashionMNIST",
                train=False,
                download=True,
                transform=transforms.Compose(
                    [
                        transforms.Pad(2),
                        transforms.ToTensor(),
                        transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
                    ]
                ),
            )
            Fashion_MNIST_train_data = np.array(
                [i.numpy() for i, _ in Fashion_MNIST_train]
            )
            Fashion_MNIST_test_data = np.array(
                [i.numpy() for i, _ in Fashion_MNIST_test]
            )

            if len(dataset) > 1:
                Fashion_MNIST_train_labels = (
                    Fashion_MNIST_train.targets.numpy() + np.max(base_labels)
                )
                Fashion_MNIST_test_labels = Fashion_MNIST_test.targets.numpy() + np.max(
                    base_labels
                )
            else:
                Fashion_MNIST_train_labels = Fashion_MNIST_train.targets.numpy()
                Fashion_MNIST_test_labels = Fashion_MNIST_test.targets.numpy()

            base_data = np.concatenate(
                [base_data, Fashion_MNIST_train_data, Fashion_MNIST_test_data]
            )
            base_labels = np.concatenate(
                [base_labels, Fashion_MNIST_train_labels, Fashion_MNIST_test_labels]
            )

    for ood_dataset in ood:
        if ood_dataset == "MNIST":

            MNIST_train = MNIST(
                root=r"/dataset/MNIST",
                train=True,
                download=True,
                transform=transforms.Compose(
                    [
                        transforms.Pad(2),
                        transforms.ToTensor(),
                        transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
                    ]
                ),
            )
            MNIST_test = MNIST(
                root=r"/dataset/MNIST",
                train=False,
                download=True,
                transform=transforms.Compose(
                    [
                        transforms.Pad(2),
                        transforms.ToTensor(),
                        transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
                    ]
                ),
            )

            MNIST_train_data = np.array([i.numpy() for i, _ in MNIST_train])
            MNIST_test_data = np.array([i.numpy() for i, _ in MNIST_test])

            MNIST_train_labels = MNIST_train.targets.numpy()
            MNIST_test_labels = MNIST_test.targets.numpy()
            OOD_data = np.concatenate([OOD_data, MNIST_train_data, MNIST_test_data])
            OOD_labels = np.concatenate(
                [OOD_labels, MNIST_train_labels, MNIST_test_labels]
            )

        elif ood_dataset == "Fashion_MNIST":

            Fashion_MNIST_train = FashionMNIST(
                root="/dataset/FashionMNIST",
                train=True,
                download=True,
                transform=transforms.Compose(
                    [
                        transforms.Pad(2),
                        transforms.ToTensor(),
                        transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
                    ]
                ),
            )
            Fashion_MNIST_test = FashionMNIST(
                root="/dataset/FashionMNIST",
                train=False,
                download=True,
                transform=transforms.Compose(
                    [
                        transforms.Pad(2),
                        transforms.ToTensor(),
                        transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
                    ]
                ),
            )
            Fashion_MNIST_train_data = np.array(
                [i.numpy() for i, _ in Fashion_MNIST_train]
            )
            Fashion_MNIST_test_data = np.array(
                [i.numpy() for i, _ in Fashion_MNIST_test]
            )
            Fashion_MNIST_train_labels = Fashion_MNIST_train.targets.numpy()
            Fashion_MNIST_test_labels = Fashion_MNIST_test.targets.numpy()

            OOD_data = np.concatenate(
                [OOD_data, Fashion_MNIST_train_data, Fashion_MNIST_test_data]
            )
            OOD_labels = np.concatenate(
                [OOD_labels, Fashion_MNIST_train_labels, Fashion_MNIST_test_labels]
            )
        elif ood_dataset == "SVHN":
            SVHN_train = SVHN(
                root=r"/dataset/SVHN",
                split="train",
                download=True,
                transform=standard_transform,
            )
            SVHN_test = SVHN(
                root=r"/dataset/SVHN",
                split="test",
                download=True,
                transform=standard_transform,
            )
            SVHN_train_data = SVHN_train.data
            SVHN_test_data = SVHN_test.data
            SVHN_train_labels = SVHN_train.labels
            SVHN_test_labels = SVHN_test.labels

            OOD_data = np.concatenate([OOD_data, SVHN_train_data, SVHN_test_data])
            OOD_labels = np.concatenate(
                [OOD_labels, SVHN_train_labels, SVHN_test_labels]
            )

        # elif ood_dataset == "TinyImageNet":
        #     if not os.listdir(os.path.join(r"./dataset/tiny-imagenet-200")):
        #         download_and_unzip()
        #     id_dict = {}
        #     for i, line in enumerate(
        #         open(
        #             os.path.join(
        #                 r"\dataset\tiny-imagenet-200\tiny-imagenet-200\wnids.txt"
        #             ),
        #             "r",
        #         )
        #     ):
        #         id_dict[line.replace("\n", "")] = i
        #     normalize_imagenet = transforms.Normalize(
        #         (122.4786, 114.2755, 101.3963), (70.4924, 68.5679, 71.8127)
        #     )
        #     train_t_imagenet = TrainTinyImageNetDataset(
        #         id=id_dict, transform=transforms.Compose([normalize_imagenet, resize])
        #     )
        #     test_t_imagenet = TestTinyImageNetDataset(
        #         id=id_dict, transform=transforms.Compose([normalize_imagenet, resize])
        #     )

    base_data = np.delete(base_data, 0, axis=0)
    base_labels = np.delete(base_labels, 0)
    OOD_data = np.delete(OOD_data, 0, axis=0)
    OOD_labels = np.delete(OOD_labels, 0)

    data_manager = Data_manager(
        base_data=base_data,
        base_labels=base_labels,
        OOD_data=OOD_data,
        OOD_labels=OOD_labels,
    )

    return data_manager
