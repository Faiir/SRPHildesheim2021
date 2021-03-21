import numpy as np
from torchvision.datasets import MNIST, FashionMNIST

# own files
from .datamanager import Data_manager


def get_datamanger():

    MNIST_train = MNIST(root=".", train=True, download=True)
    MNIST_test = MNIST(root=".", train=False, download=True)

    Fashion_MNIST_train = FashionMNIST(root=".", train=True, download=True)
    Fashion_MNIST_test = FashionMNIST(root=".", train=False, download=True)

    MNIST_train_data = MNIST_train.data.numpy()
    MNIST_test_data = MNIST_test.data.numpy()

    MNIST_train_labels = MNIST_train.targets.numpy()
    MNIST_test_labels = MNIST_test.targets.numpy()

    MNIST_all_data = np.concatenate([MNIST_train_data, MNIST_test_data])
    MNIST_all_labels = np.concatenate([MNIST_train_labels, MNIST_test_labels])

    Fashion_MNIST_train_data = Fashion_MNIST_train.data.numpy()
    Fashion_MNIST_test_data = Fashion_MNIST_test.data.numpy()

    Fashion_MNIST_train_labels = Fashion_MNIST_train.targets.numpy()
    Fashion_MNIST_test_labels = Fashion_MNIST_test.targets.numpy()

    Fashion_MNIST_all_data = np.concatenate(
        [Fashion_MNIST_train_data, Fashion_MNIST_test_data]
    )
    Fashion_MNIST_all_labels = np.concatenate(
        [Fashion_MNIST_train_labels, Fashion_MNIST_test_labels]
    )

    Data_manager = Data_manager(
        base_data=MNIST_all_data,
        base_labels=MNIST_all_labels,
        OOD_data=Fashion_MNIST_all_data,
        OOD_labels=Fashion_MNIST_all_labels,
    )

    return Data_manager