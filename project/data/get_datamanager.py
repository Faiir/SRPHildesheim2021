import numpy as np
from torchvision.datasets import MNIST, FashionMNIST

# own files
from .datamanager import Data_manager

# TODO from .datamanager import getdataset


def get_datamanager(dataset="MNIST-FMNIST"):

    if dataset == "MNIST-FMNIST":
        MNIST_train = MNIST(root=r".", train=True, download=True)
        MNIST_test = MNIST(root=r".", train=False, download=True)

        Fashion_MNIST_train = FashionMNIST(root=".", train=True, download=True)
        Fashion_MNIST_test = FashionMNIST(root=".", train=False, download=True)

        MNIST_train_data = MNIST_train.data.numpy()
        MNIST_test_data = MNIST_test.data.numpy()

        MNIST_train_labels = MNIST_train.targets.numpy()
        MNIST_test_labels = MNIST_test.targets.numpy()

        Fashion_MNIST_train_data = Fashion_MNIST_train.data.numpy()
        Fashion_MNIST_test_data = Fashion_MNIST_test.data.numpy()

        Fashion_MNIST_train_labels = Fashion_MNIST_train.targets.numpy()
        Fashion_MNIST_test_labels = Fashion_MNIST_test.targets.numpy()

        base_data = np.concatenate([MNIST_train_data, MNIST_test_data])
        base_labels = np.concatenate([MNIST_train_labels, MNIST_test_labels])

        OOD_data = np.concatenate([Fashion_MNIST_train_data, Fashion_MNIST_test_data])
        OOD_labels = np.concatenate(
            [Fashion_MNIST_train_labels, Fashion_MNIST_test_labels]
        )

    if dataset == "CHIFAR":
        base_data = "chifar"
        pass

    # TODO base_data, base_labels, OOD_data, OOD_labels = get_dataset(dataset)

    data_manager = Data_manager(
        base_data=base_data,
        base_labels=base_labels,
        OOD_data=OOD_data,
        OOD_labels=OOD_labels,
    )

    return data_manager
