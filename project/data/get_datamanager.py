import numpy as np
from torchvision.datasets import MNIST, FashionMNIST, CIFAR10, CIFAR100, SVHN

# own files
from .datamanager import Data_manager

# TODO from .datamanager import getdataset


def get_datamanager(dataset="CIFAR10-MNIST"):

    if dataset == "CIFAR10-MNIST":
        MNIST_train = MNIST(root=r".", train=True, download=True)
        MNIST_test = MNIST(root=r".", train=False, download=True)

        CIFAR10_train = CIFAR10(root=".", train=True, download=True)
        CIFAR10_test = CIFAR10(root=".", train=False, download=True)

        MNIST_train_data = MNIST_train.data.numpy()
        MNIST_test_data = MNIST_test.data.numpy()

        MNIST_train_labels = MNIST_train.targets.numpy()
        MNIST_test_labels = MNIST_test.targets.numpy()

        CIFAR10_train_data = CIFAR10_train.data.numpy()
        CIFAR10_test_data = CIFAR10_test.data.numpy()

        CIFAR10_train_labels = CIFAR10_train.targets.numpy()
        CIFAR10_test_labels = CIFAR10_test.targets.numpy()

        base_data = np.concatenate([CIFAR10_train_data, CIFAR10_test_data])
        base_labels = np.concatenate([CIFAR10_train_labels, CIFAR10_test_labels])
        
        OOD_data = np.concatenate([MNIST_train_data, MNIST_test_data])
        OOD_labels = np.concatenate([MNIST_train_labels, MNIST_test_labels])
    elif dataset == "CIFAR10-CIFAR100":
        CIFAR100_train = CIFAR100(root=r".", train=True, download=True)
        CIFAR100_test = CIFAR100(root=r".", train=False, download=True)

        CIFAR10_train = CIFAR10(root=".", train=True, download=True)
        CIFAR10_test = CIFAR10(root=".", train=False, download=True)

        CIFAR100_train_data = CIFAR100_train.data.numpy()
        CIFAR100_test_data = CIFAR100_test.data.numpy()

        CIFAR100_train_labels = CIFAR100_train.targets.numpy()
        CIFAR100_test_labels = CIFAR100_test.targets.numpy()

        CIFAR10_train_data = CIFAR10_train.data.numpy()
        CIFAR10_test_data = CIFAR10_test.data.numpy()

        CIFAR10_train_labels = CIFAR10_train.targets.numpy()
        CIFAR10_test_labels = CIFAR10_test.targets.numpy()

        base_data = np.concatenate([CIFAR10_train_data, CIFAR10_test_data])
        base_labels = np.concatenate([CIFAR10_train_labels, CIFAR10_test_labels])
        
        OOD_data = np.concatenate([CIFAR100_train_data, CIFAR100_test_data])
        OOD_labels = np.concatenate([CIFAR100_train_labels, CIFAR100_test_labels])
    elif dataset == "CIFAR10-SVHN":
        SVHN_train = SVHN(root=r".", train=True, download=True)
        SVHN_test = SVHN(root=r".", train=False, download=True)

        CIFAR10_train = CIFAR10(root=".", train=True, download=True)
        CIFAR10_test = CIFAR10(root=".", train=False, download=True)

        SVHN_train_data = SVHN_train.data.numpy()
        SVHN_test_data = SVHN_test.data.numpy()

        SVHN_train_labels = SVHN_train.targets.numpy()
        SVHN_test_labels = SVHN_test.targets.numpy()

        CIFAR10_train_data = CIFAR10_train.data.numpy()
        CIFAR10_test_data = CIFAR10_test.data.numpy()

        CIFAR10_train_labels = CIFAR10_train.targets.numpy()
        CIFAR10_test_labels = CIFAR10_test.targets.numpy()

        base_data = np.concatenate([CIFAR10_train_data, CIFAR10_test_data])
        base_labels = np.concatenate([CIFAR10_train_labels, CIFAR10_test_labels])
        
        OOD_data = np.concatenate([SVHN_train_data, SVHN_test_data])
        OOD_labels = np.concatenate([SVHN_train_labels, SVHN_test_labels])

    data_manager = Data_manager(
        base_data=base_data,
        base_labels=base_labels,
        OOD_data=OOD_data,
        OOD_labels=OOD_labels,
    )

    return data_manager
