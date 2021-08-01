from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data import random_split

from .datahandler_for_array import DataHandler_For_Arrays

def create_dataloader(current_data, batch_size=128, split_size=0.1):
    """
    Args:
        current_data: Current version of the train data and the pool to sample from
        batch_size: batch_size

    Returns:
        PyTorch's train ,val and test loader
    """
    train_X, train_y = current_data.get_train_data()
    test_X, test_y = current_data.get_test_data()
    pool_X, pool_y = current_data.get_unlabelled_pool_data()

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