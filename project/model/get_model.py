from .mnist_model import Net



def get_model(model_name):
    """get_model [function which returns instance of the experiments model]

    [extended_summary]

    Args:
        model_name ([String]): [Predefined String which is used for to instanciate the correct model]

    Raises:
        NotImplementedError: [GenOdin Model which is not Implemented yet]
        ValueError: [Unexpected Model Name]

    Returns:
        [nn.Module]: [Instanciated PyTorch Neural network]
    """
    if model_name == "base":
        net = Net()
        return net
    elif model_name == "gen_odin":
        raise NotImplementedError("This model type is not implemented yet")
    else:
        raise ValueError(f"Model {model_name} not found")