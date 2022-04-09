import argparse
import json
import os
import sys
import pandas as pd

from torch import nn, optim
import torch
import ssl

ssl._create_default_https_context = ssl._create_unverified_context
import torch

PACKAGE_PARENT = ".."
SCRIPT_DIR = os.path.dirname(
    os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__)))
)
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))


from robust_active_learning.data.data_manager import Data_manager
from robust_active_learning.data.datahandler_for_array import create_dataloader
from robust_active_learning.model.get_model import get_model

# from robust_active_learning.helpers.early_stopping import EarlyStopping
from torchsummary import summary


def verbosity(message, verbose, epoch):
    if verbose == 1:
        if epoch % 10 == 0:
            print(message)
    elif verbose == 2:
        print(message)
    return None


def train(
    train_loader,
    val_loader,
    optimizer,
    criterion,
    device,
    epochs,
    model,
    **kwargs,
):
    """train [main training function of the project]

    [extended_summary]

    Args:
        train_loader ([torch.Dataloader]): [dataloader with the training data]
        optimizer ([torch.optim]): [optimizer for the network]
        criterion ([Loss function]): [Pytorch loss function]
        device ([str]): [device to train on cpu/cuda]
        epochs (int, optional): [epochs to run]. Defaults to 5.
        **kwargs (verbose and validation dataloader)
    Returns:
        [tupel(trained network, train_loss )]:
    """
    verbose = kwargs.get("verbose", 1)

    if verbose > 0:
        print("\nTraining with device :", device)
        print("Number of Training Samples : ", len(train_loader.dataset))
        if val_loader is not None:
            print("Number of Validation Samples : ", len(val_loader.dataset))
        print("Number of Epochs : ", epochs)

        if verbose > 1:
            summary(model, input_size=(3, 32, 32))

    lr_sheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        "min",
        factor=0.1,
        patience=int(epochs * 0.05),
        min_lr=1e-7,
        verbose=True,
    )

    validation = False
    if kwargs.get("patience", None) is None:
        print(
            f"INFO ------ Early Stopping Patience not specified using {int(epochs * 0.1)}"
        )
    patience = kwargs.get("patience", int(epochs * 0.1))
    # early_stopping = EarlyStopping(patience, verbose=True, delta=1e-6)

    for epoch in range(1, epochs + 1):
        if verbose > 0:
            print(f"\nEpoch: {epoch}")

        train_loss = 0
        train_acc = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            if len(data) > 1:
                model.train()
                data, target = data.to(device).float(), target.to(device).long()

                optimizer.zero_grad(set_to_none=True)
                yhat = model(data).to(device)
                loss = criterion(yhat, target)
                train_loss += loss.item()
                train_acc += torch.sum(torch.argmax(yhat, dim=1) == target).item()

                loss.backward()
                optimizer.step()
            else:
                pass

        avg_train_loss = train_loss / len(train_loader)
        avg_train_acc = train_acc / len(train_loader.dataset)

        if epoch % 1 == 0:
            pass
            # if validation:
            #     val_loss = 0
            #     val_acc = 0
            #     model.eval()  # prep model for evaluation
            #     with torch.no_grad():
            #         for vdata, vtarget in val_loader:
            #             vdata, vtarget = (
            #                 vdata.to(device).float(),
            #                 vtarget.to(device).long(),
            #             )
            #             voutput = model(vdata)
            #             vloss = criterion(voutput, vtarget)
            #             val_loss += vloss.item()
            #             val_acc += torch.sum(
            #                 torch.argmax(voutput, dim=1) == vtarget
            #             ).item()

            #     avg_val_loss = val_loss / len(val_loader)
            #     avg_val_acc = val_acc / len(val_loader.dataset)

            #     early_stopping(avg_val_loss, model)
            #     if kwargs.get("lr_sheduler", True):
            #         lr_sheduler.step(avg_val_loss)

            #     verbosity(
            #         f"Val_loss: {avg_val_loss:.4f} Val_acc : {100*avg_val_acc:.2f}",
            #         verbose,
            #         epoch,
            #     )

            #     if early_stopping.early_stop:
            #         print(
            #             f"Early stopping epoch {epoch} , avg train_loss {avg_train_loss}, avg val loss {avg_val_loss}"
            #         )
            #         break

        verbosity(
            f"Train_loss: {avg_train_loss:.4f} Train_acc : {100*avg_train_acc:.2f}",
            verbose,
            epoch,
        )

    return model, avg_train_loss, avg_train_acc


# overrides test
@torch.no_grad()
def test(model, test_loader, device, criterion):
    """test [computes loss of the test set]

    [extended_summary]

    Returns:
        [type]: [description]
    """
    test_loss = 0
    test_acc = 0
    model.eval()
    for (t_data, t_target) in test_loader:
        t_data, t_target = (
            t_data.to(device).float(),
            t_target.to(device).long(),
        )

        t_output = model(t_data)
        t_output.to(device).long()
        t_loss = criterion(t_output, t_target)
        test_loss += t_loss
        test_acc += torch.sum(torch.argmax(t_output, dim=1) == t_target).item()

    avg_test_acc = test_acc / len(test_loader.dataset)
    avg_test_loss = test_loss.to("cpu").detach().numpy() / len(test_loader)
    return avg_test_acc, avg_test_loss


def final_training(log_dirs, config):
    for c, experiment in enumerate(config["experiments"]):
        log_dir = log_dirs[c]
        print("using logs from: ", log_dir)
        basic_settings = experiment["basic_settings"]
        # data_manager
        iD = basic_settings.get("iD", "Cifar10")
        OoD = basic_settings.get("OoD", ["Fashion_MNIST"])
        labelled_size = basic_settings.get("labelled_size", 3000)
        pool_size = basic_settings.get("pool_size", 20000)
        OOD_ratio = basic_settings.get("OOD_ratio", 0.0)
        # training settings
        epochs = 1  # 130  # basic_settings.get("epochs", 200)
        batch_size = basic_settings.get("batch_size", 128)
        weight_decay = basic_settings.get("weight_decay", 1e-4)

        lr = basic_settings.get("lr", 0.1)
        nesterov = basic_settings.get("nesterov", False)
        momentum = basic_settings.get("momentum", 0.9)
        num_classes = basic_settings.get("num_classes", 10)

        # criterion = basic_settings.get("criterion", "crossentropy")

        metric = basic_settings.get("metric", "accuracy")
        # logging
        verbose = basic_settings.get("verbose", 1)
        criterion = nn.CrossEntropyLoss()
        with open(
            os.path.join(log_dir, "final_result.csv"), "w", encoding="utf-8"
        ) as result_file:
            result_file.write(
                f"Experiment_name,Starting_size,Train_size,OOD_ratio,Train_Acc,Train_Loss,Val_Acc,Val_Loss,Test_Acc,Test_Loss\n"
            )

        subclass = basic_settings.get("subclass", {"do_subclass": False})
        with open(
            os.path.join(log_dir, "final_result.csv"), "w", encoding="utf-8"
        ) as result_file:
            result_file.write(
                "exp_name,trainsize,OOD_ratio,avg_train_acc,avg_train_loss,avg_test_acc,avg_test_loss\n"
            )
        for exp_setting in experiment["exp_settings"]:
            exp_name = exp_setting.get("exp_name", "standard_name")
            data_manager = Data_manager(
                iD_datasets=[iD],
                OoD_datasets=OoD,
                labelled_size=labelled_size,
                pool_size=pool_size,
                OoD_ratio=OOD_ratio,
                test_iD_size=None,
                subclass=subclass,
            )
            if not exp_setting.get("perform_experiment", True):
                continue
            else:
                print("performing final training for: ", exp_name)
            # data_manager.create_merged_data() TODO load the statusmanager from the path
            check_path = os.path.join(
                log_dir, "status_manager_dir", f"{exp_name}-result-statusmanager.csv"
            )
            print("loading statusmanager: ", check_path)
            if os.path.exists(check_path):
                data_manager.status_manager = pd.read_csv(check_path, index_col=0)
                # self.data_manager.reset_pool()
                data_manager.iter = 19
                print("loaded statusmanager from file")
            else:
                print("couldn't load statusmanager aborting: f{exp_name}")
                break
            result_tup = create_dataloader(
                data_manager, batch_size, 0.1, validation_source=None
            )
            train_loader = result_tup[0]
            test_loader = result_tup[1]
            # val_loader = result_tup[3]

            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

            model = get_model("base", num_classes=num_classes)
            model.to(device)
            optimizer = optim.SGD(
                model.parameters(),
                weight_decay=weight_decay,
                lr=lr,
                momentum=momentum,
                nesterov=nesterov,
            )

            if device == "cuda":
                torch.backends.cudnn.benchmark = True

            model, avg_train_loss, avg_train_acc = train(
                train_loader=train_loader,
                val_loader=None,
                optimizer=optimizer,
                criterion=criterion,
                device=device,
                epochs=epochs,
                model=model,
                verbose=verbose,
            )
            avg_test_acc, avg_test_loss = test(model, test_loader, device, criterion)

            print(
                f"""Experiment: {exp_name},
                    Final_trainingset size: {len(train_loader)},
                    OOD_ratio: {OOD_ratio},
                    Train-Accuracy: {avg_train_acc},
                    Train-Loss: {avg_train_loss},
                    Test-Accuracy: {avg_test_acc},
                    Test-Loss: {avg_test_loss}"""
            )

            with open(
                os.path.join(log_dir, "final_result.csv"), "a", encoding="utf-8"
            ) as result_file:
                result_file.write(
                    f"{exp_name},{len(train_loader.dataset)},{OOD_ratio},{avg_train_acc},{avg_train_loss},{avg_test_acc},{avg_test_loss}\n"
                )


def main():
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
        help="Log-instructions in json",
        type=str,
        default=os.path.join("./log_dirs"),
    )

    args = parser.parse_args()
    if args.config is None:
        args.config = os.path.join(".\experiment_settings.json")
    if args.log is None:
        args.config = os.path.join(".\log_dirs.json")
    config = args.config
    log_dir_info = args.log
    with open(config, mode="r", encoding="utf-8") as config_f:
        config = json.load(config_f)
    with open(log_dir_info, mode="r", encoding="utf-8") as config_f:
        log_dirs_json = json.load(config_f)

    final_training(log_dirs_json["log_dirs"], config)


if __name__ == "__main__":
    main()