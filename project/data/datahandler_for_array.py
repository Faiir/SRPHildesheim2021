## Datahandlers
from PIL import Image
from torch.utils.data import Dataset
import numpy as np

from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data import random_split
from torchvision.transforms import transforms
import torch


##setting pin_memory to False to see if the errors go away
pin_memory = False

class DataHandler_For_Arrays(Dataset):
    """DataHandler_For_Arrays [Base pytorch dataset for all experiments]"""

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


def create_dataloader(
    data_manager, batch_size=128, validation_split=None, validation_source=None
):
    """
    Args:
        data_manager: Current version of the train data and the pool to sample from
        batch_size: batch_size
        validation_source : Whether to use train or test dataset for creating validation dataset, Defauts to None
        validation_split : Specify the ratio of samples to use in validation, defaults to 20% of source size

    Returns:
        PyTorch's train, test and pool loader. (Validation loader is also returned if source is not None)
    """
    pool_dataset = data_manager.get_unlabelled_pool_dataset()
    train_dataset = data_manager.get_train_dataset()
    test_dataset = data_manager.get_test_dataset()

    if validation_source is None:
        print(
            "INFO ------ Validation source not specified in config, experiment would run without validation set"
        )
    else:
        if validation_source == "test":
            if validation_split is None:
                validation_size = int(len(test_dataset) * 0.2)
            else:
                assert (
                    0 < validation_split < 1
                ), f"Validation size must be >0 and <1, found {validation_split}"
                validation_size = int(len(test_dataset) * validation_split)

            print(
                f"Using Testing data to create validation dataset, size : {validation_size}"
            )
            test_dataset, validation_dataset = random_split(
                test_dataset,
                lengths=[len(test_dataset) - validation_size, validation_size],
            )
        elif validation_source == "train":
            if validation_split is None:
                validation_size = int(len(train_dataset) * 0.2)
            else:
                assert (
                    0 < validation_split < 1
                ), f"Validation size must be >0 and <1, found {validation_split}"
                validation_size = int(len(train_dataset) * validation_split)

            print(
                f"Using Training data to create validation dataset, size : {validation_size}"
            )
            train_dataset, validation_dataset = random_split(
                train_dataset,
                lengths=[len(train_dataset) - validation_size, validation_size],
            )

    train_loader = DataLoader(
        train_dataset,
        sampler=RandomSampler(train_dataset),
        batch_size=batch_size,
        num_workers=2,
        pin_memory=pin_memory,
    )

    test_loader = DataLoader(
        test_dataset,
        sampler=SequentialSampler(test_dataset),
        batch_size=batch_size,
        num_workers=2,
        pin_memory=pin_memory,
    )

    pool_loader = DataLoader(
        pool_dataset,
        sampler=SequentialSampler(pool_dataset), 
        batch_size=batch_size, 
        num_workers=2, 
        pin_memory=pin_memory
    )

    if validation_source is not None:
        val_loader = DataLoader(
            validation_dataset,
            sampler=SequentialSampler(validation_dataset),
            batch_size=batch_size,
            num_workers=2,
            pin_memory=pin_memory,
            )


        return (train_loader, test_loader, pool_loader, val_loader)

    else:
        return (train_loader, test_loader, pool_loader)



def get_ood_dataloader(data_manager, batch_size: int = 16):
    """get_ood_dataloader [Returns OOD dataloader for Outlier exposure experiment]
    Args:
        data_manager ([object]): [datamanager]
        batch_size (int, optional): [batch size for dataloader]. Defaults to 16.

    Returns:
        [torch.Dataloader]: [train loader]
        [torch.Dataloader]: [ood lloader]
    """
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
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.RandomCrop(size=32),
            transforms.RandomRotation(degrees=(0, 180)),
            transforms.RandomHorizontalFlip(),
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
        pin_memory=pin_memory,
    )

    outlier_loader = DataLoader(
        outlier_data,
        sampler=RandomSampler(outlier_data),
        batch_size=batch_size,
        num_workers=2,
        pin_memory=pin_memory,
    )
    return train_loader, outlier_loader
