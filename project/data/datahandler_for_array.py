## Datahandlers
from PIL import Image
from torch.utils.data import Dataset
import numpy as np

from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data import random_split


class DataHandler_For_Arrays(Dataset):
    """
    base class for mnist / fashion_mnist
    """

    def __init__(self, X, Y, transform=None, num_classes=10):
        self.X = X  # X[np.newaxis,...] # x[:, np.newaxis]:
        self.Y = Y
        # self.Y = torch.as_tensor(self.Y)
        # self.Y = torch.nn.functional.one_hot(self.Y, num_classes=10)
        self.transform = transform
        self.num_classes = num_classes

    def __getitem__(self, index):
        x, y = self.X[index], self.Y[index]

        x = x / 255.0
        if self.transform is not None:
            x = Image.fromarray(x.numpy(), mode="L")
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.X)


def create_dataloader(data_manager, batch_size=128, split_size=0.1):
    """
    Args:
        data_manager: Current version of the train data and the pool to sample from
        batch_size: batch_size

    Returns:
        PyTorch's train ,val and test loader
    """
    train_X, train_y = data_manager.get_train_data()
    test_X, test_y = data_manager.get_test_data()
    pool_X, pool_y = data_manager.get_unlabelled_pool_data()

    pool_dataset = DataHandler_For_Arrays(pool_X, pool_y)

    train_dataset = DataHandler_For_Arrays(train_X, train_y)

    test_dataset = DataHandler_For_Arrays(test_X, test_y)

    train_loader = DataLoader(
        train_dataset, sampler=RandomSampler(train_dataset), batch_size=batch_size
    )

    test_loader = DataLoader(
        test_dataset, sampler=SequentialSampler(test_dataset), batch_size=batch_size
    )

    pool_loader = DataLoader(pool_dataset, batch_size=batch_size)

    return train_loader, test_loader, pool_loader  # , train_dataset, test_dataset


def get_dataloader(data_manager, batch_size=128, split_size=0.1):

    train_loader, test_loader, pool_loader = create_dataloader(
        data_manager, batch_size=128, split_size=0.1
    )

    return train_loader, test_loader, pool_loader