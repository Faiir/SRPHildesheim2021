import json


def start_experiment(config_path, log):
    # print(config_path)
    # print(log)
    log_dir = log
    config = ""

    with open(config_path, mode="r", encoding="utf-8") as config_f:
        config = json.load(config_f)

    print(
        """
    **********************************************
        EXPERIMENT DONE
    **********************************************
    """
    )

    # initalize the model
    # init