from .model_files.mnist_model import Net
from .model_files.genOdinModel import genOdinModel
from .model_files.big_resnet import resnet18
from .model_files.small_resnet__wPTSpec import resnet20 as resnet20SpecGenOdin
from .model_files.small_resnet_original import resnet20 as resnet20_original
from .model_files.small_resnet_only_specnorm import (
    resnet20 as resnet20_original_spec_norm,
)
from .model_files.small_resnet import resnet20


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
    num_classes=10,
    include_bn=False,
    channel_input=3,
    **kwargs,
):
    """get_model [[function which returns instance of the experiments model]]

    [extended_summary]

    Args:
        model_name ([string]): ["base":conv_net, "gen_odin_conv":conv_net with GenOdin,
            "gen_odin_res": large resnet with GenOdin Layer, "small_gen_odin_res": small resnet with GenOdin Layer,
            "small_resnet_with_spec": small resnet with spectral normalization, "base_small_resnet": abdurs resnet (working the best as per usual)  ]
        similarity ([type], optional): [For genOdinMode "E":Euclidean distance, "I": FC Layer, "C": Cosine Similarity, "IR": I-reversed , "ER": E-revesered, "CR": "C-reversed" ]. Defaults to None.
        num_classes (int, optional): [Number of classes]. Defaults to 10.
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
            out_classes=num_classes,
            include_bn=include_bn,
            channel_input=channel_input,
        )
        return genOdin
    elif model_name == "gen_odin_res":
        return resnet18(
            similarity=similarity, do_not_genOdin=kwargs.get("do_not_genOdin", False)
        )
    elif model_name == "small_gen_odin_res":
        return resnet20(
            similarity=similarity,
            selfsupervision=kwargs.get("selfsupervision", False),
            num_classes=num_classes,
            do_not_genOdin=kwargs.get("do_not_genOdin", False),
        )

    elif model_name == "small_resnet_with_spec":
        return resnet20SpecGenOdin(
            similarity=similarity,
            selfsupervision=kwargs.get("selfsupervision", False),
            num_classes=num_classes,
            do_not_genOdin=kwargs.get("do_not_genOdin", False),
        )

    elif model_name == "base_small_resnet":
        if num_classes!=10:
            print(f'INFO ---- Number of classes is {num_classes}')
        return resnet20_original(
            num_classes=num_classes, similarity=similarity
        )

    else:
        raise ValueError(f"Model {model_name} not found")


def save_model(
    net,
    optimizer,
    exp_conf,
    datamanager,
    path,
    in_dist,
    ood_data,
    desc_str="pretrained_net",
):
    time = datetime.now().strftime("%d-%m-%H")
    path = os.path.join(path, desc_str)
    status_manager_copy = datamanager.status_manager.copy()
    torch.save(
        {
            "model_state_dict": net.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "experiment config": exp_conf,
            "status_manager": status_manager_copy,
            "in_dist": in_dist,
            "ood_data": ood_data,
        },
        path + ".pt",
    )


# def remove_rot_heads(net):
#     # https://stackoverflow.com/questions/52548174/how-to-remove-the-last-fc-layer-from-a-resnet-model-in-pytorch
#     # TODO
#     return net
