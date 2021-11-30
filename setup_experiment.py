import argparse
import gc
import sys
import os

import torch

PACKAGE_PARENT = ".."
SCRIPT_DIR = os.path.dirname(
    os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__)))
)
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

import json
from torch.utils.tensorboard import SummaryWriter
import torch.backends.cudnn as cudnn

from robust_active_learning.experiment_active_learning import experiment_active_learning
from robust_active_learning.experiment_ddu import experiment_ddu
from robust_active_learning.experiment_genOdin import experiment_gen_odin
from robust_active_learning.experiment_gram import experiment_gram
from robust_active_learning.experiment_extraclass import experiment_extraclass
from robust_active_learning.experiment_without_OoD import experiment_without_OoD
from robust_active_learning.helpers.final_train import final_traing

# import shutil
import time

final_training = False


def create_log_dirs(log_path):
    if os.path.exists(log_path) == False:
        os.makedirs(log_path)

    status_manager_path = os.path.join(log_path, "status_manager_dir")
    writer_path = os.path.join(log_path, "writer_dir")
    log_dir_path = os.path.join(log_path, "log_dir")

    if os.path.exists(status_manager_path) == False:
        os.mkdir(status_manager_path)
    if os.path.exists(writer_path) == False:
        os.mkdir(writer_path)
    if os.path.exists(log_dir_path) == False:
        os.mkdir(log_dir_path)

    print("Directories created")


def start_experiment(config, log_path):
    if log_path == os.path.join("./logs"):
        log_path = os.path.join(
            log_path, time.strftime("%m-%d-%H-%M", time.localtime())
        )

    print("Logging Results under: ", log_path)

    create_log_dirs(log_path)

    writer = SummaryWriter(os.path.join(log_path, "writer_dir"))

    if torch.cuda.is_available():
        cudnn.benchmark = True

    with open(config, mode="r", encoding="utf-8") as config_f:
        config = json.load(config_f)

    for experiment in config["experiments"]:
        basic_settings = experiment["basic_settings"]

        for exp_setting in experiment["exp_settings"]:
            if exp_setting.get("perform_experiment", True):
                print(
                    f'\n\nINFO ---- Experiment {exp_setting["exp_type"]} is being performed.\n\n'
                )
            else:
                print(
                    f'\n\nINFO ---- Experiment {exp_setting["exp_type"]} is not being performed.\n\n'
                )
                continue

            exp_type = exp_setting["exp_type"]

            if exp_type == "baseline":
                current_exp = experiment_without_OoD(
                    basic_settings, exp_setting, log_path, writer
                )

            elif exp_type == "baseline-ood":
                current_exp = experiment_active_learning(
                    basic_settings, exp_setting, log_path, writer
                )
            elif exp_type == "extra_class":
                current_exp = experiment_extraclass(
                    basic_settings, exp_setting, log_path, writer
                )
            elif exp_type == "gram":
                current_exp = experiment_gram(
                    basic_settings, exp_setting, log_path, writer
                )
            elif exp_type == "looc":
                current_exp = experiment_gen_odin(
                    basic_settings, exp_setting, log_path, writer
                )
            elif exp_type == "genodin":
                current_exp = experiment_gen_odin(
                    basic_settings, exp_setting, log_path, writer
                )
            elif exp_type == "ddu":
                current_exp = experiment_ddu(
                    basic_settings, exp_setting, log_path, writer
                )

            current_exp.perform_experiment()
            del current_exp
            gc.collect()
    if final_traing:
        print("performing final training on the datamanagers")
        final_traing(log_path, config)


def main():
    """main [main function which is the entry point of this python project, takes command line arguments and sends them to the experiment setup file]

    [extended_summary]
    """

    # Initiate argument parser
    parser = argparse.ArgumentParser(
        description="Preare run of AL with OoD experiment",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "-c",
        "--config",
        help="Path to the config file for the experiment",
        type=str,
        default=None,
    )
    parser.add_argument(
        "-l",
        "--log",
        help="Log folder",
        type=str,
        default=os.path.join("./logs"),
    )
    args = parser.parse_args()

    if args.config is None:
        args.config = os.path.join(".\exp-config.json")

    start_experiment(args.config, args.log)


if __name__ == "__main__":
    main()
