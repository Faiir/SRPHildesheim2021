from .mnist_model import Net
from .genOdinModel import genOdinModel
from .resnet import resnet18


def get_model(
    model_name,
    similarity=None,
    out_classes=10,
    include_bn=False,
    channel_input=3,
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
    else:
        raise ValueError(f"Model {model_name} not found")