## Datahandlers
from PIL import Image
from torch.utils.data import Dataset
import numpy as np

from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data import random_split
from torchvision.transforms import transforms
import torch


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

        if self.transform:
            x = self.transform(x)

        return x, y

    def __len__(self):
        return len(self.X)


def create_dataloader(data_manager, batch_size=128):
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

    train_X, train_y = train_X.astype(np.float32), train_y.astype(np.float32)
    test_X, test_y = test_X.astype(np.float32), test_y.astype(np.float32)
    pool_X, pool_y = pool_X.astype(np.float32), pool_y.astype(np.float32)

    train_X, train_y = torch.from_numpy(train_X), torch.from_numpy(train_y)
    test_X, test_y = torch.from_numpy(test_X), torch.from_numpy(test_y)
    pool_X, pool_y = torch.from_numpy(pool_X), torch.from_numpy(pool_y)

#    transform_train = transforms.Compose(
#        [
#            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
#            transforms.ColorJitter(
#                brightness=(0.25, 0.75),
#                contrast=(0.25, 0.75),
#                saturation=(0.25, 0.75),
#                hue=(-0.25, 0.25),
#            ),
#            transforms.RandomHorizontalFlip(),
#           transforms.RandomCrop(size=32),
#            transforms.RandomRotation(degrees=(0, 180)),
#        ]
#    )

    transform_train = transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
        ])

    # Normalize the test set same as training set without augmentation
#    transform_test = transforms.Compose(
#        [
#            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
#        ]
#    )

    transform_test = transforms.Compose([
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
        ])

    pool_dataset = DataHandler_For_Arrays(pool_X, pool_y)

    train_dataset = DataHandler_For_Arrays(train_X, train_y, transform=transform_train)

    test_dataset = DataHandler_For_Arrays(test_X, test_y, transform=transform_test)

    train_loader = DataLoader(
        train_dataset,
        sampler=RandomSampler(train_dataset),
        batch_size=batch_size,
        num_workers=2,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        sampler=SequentialSampler(test_dataset),
        batch_size=batch_size,
        num_workers=2,
        pin_memory=True,
    )

    pool_loader = DataLoader(
        pool_dataset, batch_size=batch_size, num_workers=2, pin_memory=True
    )

    return train_loader, test_loader, pool_loader  # , train_dataset, test_dataset


def create_dataloader_with_validation(data_manager, batch_size=128):
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

    train_X, train_y = train_X.astype(np.float32), train_y.astype(np.float32)
    test_X, test_y = test_X.astype(np.float32), test_y.astype(np.float32)
    pool_X, pool_y = pool_X.astype(np.float32), pool_y.astype(np.float32)

    train_X, train_y = torch.from_numpy(train_X), torch.from_numpy(train_y)
    test_X, test_y = torch.from_numpy(test_X), torch.from_numpy(test_y)
    pool_X, pool_y = torch.from_numpy(pool_X), torch.from_numpy(pool_y)

    transform_train = transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
        ])

    # Normalize the test set same as training set without augmentation
    transform_test = transforms.Compose([
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
        ])

    pool_dataset = DataHandler_For_Arrays(pool_X, pool_y)

    train_dataset = DataHandler_For_Arrays(train_X, train_y, transform=transform_train)

    test_dataset = DataHandler_For_Arrays(test_X, test_y, transform=transform_test)

    test_dataset, validation_dataset = random_split(
        test_dataset,
        lengths=[int(len(test_dataset) * 0.5), int(len(test_dataset) * 0.5)],
    )

    train_loader = DataLoader(
        train_dataset,
        sampler=RandomSampler(train_dataset),
        batch_size=batch_size,
        num_workers=2,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        sampler=SequentialSampler(test_dataset),
        batch_size=batch_size,
        num_workers=2,
        pin_memory=True,
    )

    val_loader = DataLoader(
        validation_dataset,
        sampler=SequentialSampler(validation_dataset),
        batch_size=batch_size,
        num_workers=2,
        pin_memory=True,
    )

    pool_loader = DataLoader(
        pool_dataset, batch_size=batch_size, num_workers=2, pin_memory=True
    )

    return (
        train_loader,
        test_loader,
        val_loader,
        pool_loader,
    )  # , train_dataset, test_dataset


def get_dataloader(data_manager, batch_size=128):

    train_loader, test_loader, pool_loader = create_dataloader(
        data_manager, batch_size=128
    )

    return train_loader, test_loader, pool_loader


def get_ood_dataloader(data_manager, batch_size=16):
    train_X, train_y = data_manager.get_train_data()
    ood_X, ood_y = data_manager.get_ood_data()

    train_y = np.ones_like(train_y)
    ood_y = np.zeros_like(ood_y)

    train_X, train_y = train_X.astype(np.float32), train_y.astype(np.float32)
    ood_X, ood_y = ood_X.astype(np.float32), ood_y.astype(np.float32)

    train_X, train_y = torch.from_numpy(train_X), torch.from_numpy(train_y)
    ood_X, ood_y = torch.from_numpy(ood_X), torch.from_numpy(ood_y)

    train_dataset = DataHandler_For_Arrays(train_X, train_y)

    transform_ood = transforms.Compose(
        [
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            transforms.ColorJitter(
                brightness=(0.25, 0.75),
                contrast=(0.25, 0.75),
                saturation=(0.25, 0.75),
                hue=(-0.25, 0.25),
            ),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=32),
            transforms.RandomRotation(degrees=(0, 180)),
        ]
    )

    outlier_data = DataHandler_For_Arrays(
        train_X, train_y, transform=transform_ood, num_classes=1
    )

    train_loader = DataLoader(
        train_dataset,
        sampler=RandomSampler(train_dataset),
        batch_size=batch_size * 4,
        num_workers=2,
        pin_memory=True,
    )

    outlier_loader = DataLoader(
        outlier_data,
        sampler=RandomSampler(outlier_data),
        batch_size=batch_size,
        num_workers=2,
        pin_memory=True,
    )
    return train_loader, outlier_loader
