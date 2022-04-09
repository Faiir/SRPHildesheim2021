import argparse
import gc
import sys
import os
import ssl

ssl._create_default_https_context = ssl._create_unverified_context
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
from robust_active_learning.final_train import final_training

# import shutil
import time

final_training_sett = False


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
    base_log_path = log_path
    if torch.cuda.is_available():
        cudnn.benchmark = True

    with open(config, mode="r", encoding="utf-8") as config_f:
        config = json.load(config_f)

    for experiment in config["experiments"]:
        if base_log_path == os.path.join("./logs"):
            log_path = os.path.join(
                log_path, time.strftime("%m-%d-%H-%M", time.localtime())
            )
        print("Logging Results under: ", log_path)
        create_log_dirs(log_path)
        try:
            if final_training_sett == True:
                with open(
                    os.path.join(os.getcwd(), "log_dirs.json"),
                    mode="r+",
                    encoding="utf-8",
                ) as log_json:
                    final_training_logs = json.load(log_json)
                    final_training_logs["log_dirs"].append(log_path)
                    log_json.seek(0)  # rewind
                    json.dump(final_training_logs, log_json)
                    print("added logdir")
                    log_json.truncate()
        except:
            pass
        writer = SummaryWriter(os.path.join(log_path, "writer_dir"))

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
            try:
                current_exp.perform_experiment()
                current_exp
                gc.collect()

            except Exception as e:
                name = exp_setting["exp_name"]
                print("\n\n")
                print("**********" * 12)
                print(f"Experiment {name} failed with Exception {e}")
                print("**********" * 12)
                print("\n\n")

        log_path = base_log_path
    # final_training_sett = False
    if final_training_sett:
        print("performing final training on the data_managers")
        try:
            with open(
                os.path.join(os.getcwd(), "log_dirs.json"),
                mode="r+",
                encoding="utf-8",
            ) as log_json:
                final_training_logs = json.load(log_json)
            final_training(final_training_logs["log_dirs"], config)
        except:
            print("final training failed")


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
        args.config = os.path.join(".\experiment_settings.json")

    start_experiment(args.config, args.log)


if __name__ == "__main__":
    main()
