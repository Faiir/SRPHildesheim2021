import os
import argparse
import sys
import os

PACKAGE_PARENT = ".."
SCRIPT_DIR = os.path.dirname(
    os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__)))
)
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))


# import shutil
from project.experiment import start_experiment


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
