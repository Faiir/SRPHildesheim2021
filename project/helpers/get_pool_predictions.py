import numpy as np
import torch


@torch.no_grad()
def get_pool_predictions(trained_net, pool_loader, device, return_labels=False):
    """get_pool_predictions [predictions for the unlabelled pool]

    [extended_summary]

    Args:
        trained_net ([nn.Module]): [description]
        pool_loader ([Dataloader]): [description]
        device ([str]): [description]
        return_labels (bool, optional): [return the labels]. Defaults to False.

    Returns:
        [type]: [description]
    """
    trained_net.eval()

    yhat = []
    labels_list = []
    for (data, labels) in pool_loader:
        pred = trained_net(data.to(device).float())
        yhat.append(pred.to("cpu").detach().numpy())
        labels_list.append(labels)

    predictions = np.concatenate(yhat)
    labels_list = np.concatenate(labels_list)

    if return_labels:
        return predictions, labels_list
    else:
        return predictions
