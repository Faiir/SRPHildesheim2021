import os

from ..data.datamanager import Data_manager
from ..data.datahandler_for_array import create_dataloader
from ..model.get_model import get_model


def final_traing(log_dir, config):
    status_manager_dir = os.path.join(log_dir, "status_manager_dir")
    result_dir = os.path.join(log_dir, "results")

    for experiment in config["experiments"]:
        basic_settings = experiment["basic_settings"]
        for exp_setting in experiment["exp_settings"]:
            pass
