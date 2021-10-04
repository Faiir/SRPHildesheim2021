import numpy as np


def random_sample(dataset_manager, number_samples, net, predictions=None):
    """random_sample [Randomly adds images from the unlabelled pool to the training set]
    Args:
        dataset_manager ([object]): [description]
        number_samples ([int]): [oracle stepsize]
        net ([nn.Module]): [description]
        predictions ([type], optional): [description]. Defaults to None.
    """

    status_manager = dataset_manager.status_manager
    pool_samples_count = len(status_manager[status_manager["status"] == 0])

    assert pool_samples_count > 0, "No sample left in pool to label"
    assert (
        pool_samples_count > number_samples
    ), f"Number of samples to be labelled is less than the number of samples left in pool : {pool_samples_count} < {number_samples}"

    inds = np.random.choice(
        status_manager[status_manager["status"] == 0].index.tolist(),
        replace=False,
        size=number_samples,
    )
    iteration = 1 + status_manager["status"].max()
    status_manager.iloc[inds, -1] = iteration * status_manager.iloc[inds, -2]

    return None


def uncertainity_sampling_least_confident(
    dataset_manager, number_samples, net, predictions=None
):
    """uncertainity_sampling_least_confident [Uses least confidence to sample training data from the unlabelled poo]

        [This function selects num_samples from the pool of  all the unlabelled data at random and add them to labelled training data assumes prediction is in shape (number_of_samples,num_classes)
    ]

        Args:
            dataset_manager ([object]): [description]
            number_samples ([int]): [oracle stepsize]
            net ([nn.Module]): [description]
            predictions ([type], optional): [description]. Defaults to None.
    """

    status_manager = dataset_manager.status_manager
    pool_samples_count = len(status_manager[status_manager["status"] == 0])

    assert pool_samples_count > 0, "No sample left in pool to label"
    assert (
        pool_samples_count > number_samples
    ), f"Number of samples to be labelled is less than the number of samples left in pool : {pool_samples_count} < {number_samples}"

    inds = np.argsort(np.max(predictions, axis=1))[:number_samples]
    inds = status_manager[status_manager["status"] == 0].index[inds]
    iteration = 1 + status_manager["status"].max()
    status_manager.iloc[inds, -1] = iteration * status_manager.iloc[inds, -2]

    return None


def uncertainity_sampling_highest_entropy(
    dataset_manager, number_samples, net, predictions=None
):
    """uncertainity_sampling_highest_entropy [Uses highest entropy to sample training data from the unlabelled pool]

    [This function selects num_samples from the pool of  all the unlabelled data at random and add them to labelled training data assumes prediction is in shape (number_of_samples,num_classes)]

    Args:
        dataset_manager ([type]): [description]
        number_samples ([type]): [description]
        net ([type]): [description]
        predictions ([type], optional): [description]. Defaults to None.
    """

    status_manager = dataset_manager.status_manager
    pool_samples_count = len(status_manager[status_manager["status"] == 0])

    assert pool_samples_count > 0, "No sample left in pool to label"
    assert (
        pool_samples_count > number_samples
    ), f"Number of samples to be labelled is less than the number of samples left in pool : {pool_samples_count} < {number_samples}"

    entropy = np.sum(predictions * np.log(predictions + 1e-9), axis=1)
    inds = np.argsort(entropy)[:number_samples]
    inds = status_manager[status_manager["status"] == 0].index[inds]
    iteration = 1 + status_manager["status"].max()
    status_manager.iloc[inds, -1] = iteration * status_manager.iloc[inds, -2]

    return None


def gen0din_sampler(dataset_manager, number_samples, net, predictions=None):

    return None


def DDU_sampler(dataset_manager, number_samples, net, predictions=None):

    return None