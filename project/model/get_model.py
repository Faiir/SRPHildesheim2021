from .mnist_model import Net
from .genOdinModel import genOdinModel
from .resnet import resnet18
from .small_resnet import resnet20
from datetime import datetime
import torch
import os
from torch import nn


def add_rot_heads(net, pernumile_layer_size=128):
    net.x_trans_head = nn.Linear(pernumile_layer_size, 3)
    net.y_trans_head = nn.Linear(pernumile_layer_size, 3)
    net.rot_head = nn.Linear(pernumile_layer_size, 4)

    return net


def get_model(
    model_name,
    similarity=None,
    out_classes=10,
    include_bn=False,
    channel_input=3,
    **kwargs,
):
    """get_model [[function which returns instance of the experiments model]]

    [extended_summary]

    Args:
        model_name ([string]): ["base":conv_net, "gen_odin_conv":conv_net with GenOdin, ]
        similarity ([type], optional): [For genOdinMode "E":Euclidean distance, "I": , "C": Cosine Similarity]. Defaults to None.
        out_classes (int, optional): [Number of classes]. Defaults to 10.
        include_bn (bool, optional): [Include batchnorm]. Defaults to False.
        channel_input (int, optional): [dataset channel]. Defaults to 3.

    Raises:
        ValueError: [When model name false]

    Returns:
        [nn.Module]: [parametrized Neural Network]
    """
    if model_name == "base":
        net = Net()
        return net
    elif model_name == "gen_odin_conv":
        genOdin = genOdinModel(
            similarity=similarity,
            out_classes=out_classes,
            include_bn=include_bn,
            channel_input=channel_input,
        )
        return genOdin
    elif model_name == "gen_odin_res":
        return resnet18(similarity=similarity)
    elif model_name == "small_gen_odin_res":
        return resnet20(
            similarity=similarity,
            selfsupervision=kwargs.get("selfsupervision", False),
            num_classes=kwargs.get("num_classes", 10),
        )
    else:
        raise ValueError(f"Model {model_name} not found")


def save_model(net, path, desc_str="pretrained_net"):
    time = datetime.now().strftime("%d-%m-%Y-%H:%M:%S")
    torch.save(net.state_dict(), os.path.join(path, desc_str, time))


def remove_rot_heads(net):
    # https://stackoverflow.com/questions/52548174/how-to-remove-the-last-fc-layer-from-a-resnet-model-in-pytorch
    # TODO
    return net