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
    weighting_factor_list = []
    labels_list = []
    for (data, labels) in pool_loader:
        if trained_net.has_weighing_factor:
            tuple_data = trained_net(data.to(device).float(), get_test_model=True)
            pred = tuple_data[0]
            weighting_factor = tuple_data[1]
            weighting_factor_list.append(weighting_factor.to("cpu").detach().numpy())
        else:
            pred = trained_net(data.to(device).float())
               
        yhat.append(pred.to("cpu").detach().numpy())
        labels_list.append(labels)

    predictions = np.concatenate(yhat)
    if len(weighting_factor_list)>0:
        weighting_factor_list = np.concatenate(weighting_factor_list)
    ## checking if the results are normalized or not. If not then applying softmax
    if not allclose(predictions.sum(axis=1),1):
        predictions = np.exp(predictions) / np.sum(np.exp(predictions), axis=0)
        
    labels_list = np.concatenate(labels_list)

    if return_labels:
        return predictions, labels_list, weighting_factor_list
    else:
        return predictions, weighting_factor_list
