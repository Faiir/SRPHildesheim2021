from typing import Dict, List, Union

# python
import datetime
import os
import json
import numpy as np


import pandas as pd

from torch.utils.tensorboard.writer import SummaryWriter
from tqdm import tqdm

# torch
import torch
import torch.nn as nn
import torch.optim as optim
from torchsummary import summary

from .helpers.measures import accuracy, auroc, f1


# project
from .experiment_base import experiment_base
from .model.get_model import get_model
from .helpers.early_stopping import EarlyStopping
from .helpers.plots import get_tsne_plot
from .data.datamanager import Data_manager
from .data.datahandler_for_array import create_dataloader


def verbosity(message, verbose, epoch):
    if verbose == 1:
        if epoch % 10 == 0:
            print(message)
    elif verbose == 2:
        print(message)
    return None


def _create_log_path_al(log_dir: str = ".", OOD_ratio: float = 0.0) -> None:
    current_time = datetime.now().strftime("%H-%M-%S")
    log_file_name = "Experiment-from-" + str(current_time) + ".csv"
    current_time = datetime.now().strftime("%H-%M-%S")
    log_file_name = (
        "Experiment-from-"
        + str(current_time)
        + "-"
        + "stanard-al"
        + "-"
        + str(OOD_ratio)
    )

    log_dir = os.path.join(".", "log_dir")

    if os.path.exists(log_dir) == False:
        os.mkdir(os.path.join(".", "log_dir"))

    log_path = os.path.join(log_dir, log_file_name)

    return log_path


class experiment_active_learning(experiment_base):
    def __init__(
        self,
        basic_settings: Dict,
        exp_settings: Dict,
        log_path: str,
        writer: SummaryWriter,
    ) -> None:
        super().__init__(basic_settings, exp_settings, log_path, writer)
        self.log_path = log_path
        self.writer = writer
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        basic_settings.update(exp_settings)
        self.current_experiment = basic_settings

        self.load_settings()

        if self.device == "cuda":
            torch.backends.cudnn.benchmark = True

        self.construct_datamanager()

    # overrides train
    def train(self, train_loader, val_loader, optimizer, criterion, device, **kwargs):
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
        # verbose = kwargs.get("verbose", 1)

        if self.verbose > 0:
            print("\nTraining with device :", device)
            print("Number of Training Samples : ", len(train_loader.dataset))
            if val_loader is not None:
                print("Number of Validation Samples : ", len(val_loader.dataset))
            print("Number of Epochs : ", self.epochs)

            if self.verbose > 1:
                summary(self.model, input_size=(3, 32, 32))

        lr_sheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            "min",
            factor=0.1,
            patience=int(self.epochs * 0.05),
            min_lr=1e-7,
            verbose=True,
        )

        validation = True
        if kwargs.get("patience", None) is None:
            print(
                f"INFO ------ Early Stopping Patience not specified using {int(self.epochs * 0.1)}"
            )
        patience = kwargs.get("patience", int(self.epochs * 0.1))
        early_stopping = EarlyStopping(patience, verbose=True, delta=1e-6)

        for epoch in tqdm(range(1, self.epochs + 1)):
            if self.verbose > 0:
                print(f"\nEpoch: {epoch}")

            train_loss = 0
            train_acc = 0
            for batch_idx, (data, target) in enumerate(train_loader):
                if len(data) > 1:
                    self.model.train()
                    data, target = data.to(device).float(), target.to(device).long()

                    optimizer.zero_grad(set_to_none=True)
                    yhat = self.model(data).to(device)
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
                if validation:
                    val_loss = 0
                    val_acc = 0
                    self.model.eval()  # prep self.model for evaluation
                    with torch.no_grad():
                        for vdata, vtarget in val_loader:
                            vdata, vtarget = (
                                vdata.to(device).float(),
                                vtarget.to(device).long(),
                            )
                            voutput = self.model(vdata)
                            vloss = criterion(voutput, vtarget)
                            val_loss += vloss.item()
                            val_acc += torch.sum(
                                torch.argmax(voutput, dim=1) == vtarget
                            ).item()

                    avg_val_loss = val_loss / len(self.val_loader)
                    avg_val_acc = val_acc / len(self.val_loader.dataset)

                    early_stopping(avg_val_loss, self.model)
                    if kwargs.get("lr_sheduler", True):
                        lr_sheduler.step(avg_val_loss)

                    verbosity(
                        f"Val_loss: {avg_val_loss:.4f} Val_acc : {100*avg_val_acc:.2f}",
                        self.verbose,
                        epoch,
                    )

                    if early_stopping.early_stop:
                        print(
                            f"Early stopping epoch {epoch} , avg train_loss {avg_train_loss}, avg val loss {avg_val_loss}"
                        )
                        break

            verbosity(
                f"Train_loss: {avg_train_loss:.4f} Train_acc : {100*avg_train_acc:.2f}",
                self.verbose,
                epoch,
            )

        self.avg_train_loss_hist = avg_train_loss
        self.avg_val_loss_hist = avg_val_loss
        self.avg_train_acc_hist = avg_train_acc
        self.avg_val_loss_hist = avg_val_acc

    # overrides test
    @torch.no_grad()
    def test(self):
        """test [computes loss of the test set]

        [extended_summary]

        Returns:
            [type]: [description]
        """
        test_loss = 0
        test_acc = 0
        self.model.eval()
        for (t_data, t_target) in self.test_loader:
            t_data, t_target = (
                t_data.to(self.device).float(),
                t_target.to(self.device).long(),
            )

            t_output = self.model(t_data)
            t_output.to(self.device).long()
            t_loss = self.criterion(t_output, t_target)
            test_loss += t_loss
            test_acc += torch.sum(torch.argmax(t_output, dim=1) == t_target).item()

        self.avg_test_acc = test_acc / len(self.test_loader.dataset)
        self.avg_test_loss = test_loss.to("cpu").detach().numpy() / len(
            self.test_loader
        )  # return avg testloss

    # overrides save_settings
    def save_settings(self) -> None:
        log_config_path = os.path.join(self.log_dir + ".json")
        with open(log_config_path, "w") as f:
            json.dump(self.current_experiment, f)

    def save_al_logs(self) -> None:
        log_df = self.datamanager.get_logs()
        al_logs = os.path.join(self.log_path, "log_dir", f"logs-{self.exp_name}.csv")
        with open(al_logs, mode="w", encoding="utf-8") as logfile:
            colums = log_df.columns
            for colum in colums:
                logfile.write(colum + ",")
            logfile.write("\n")
            for _, row in log_df.iterrows():
                for c in colums:
                    logfile.write(str(row[c].item()))
                    logfile.write(",")
                logfile.write("\n")

    # overrides construct_datamanager
    def construct_datamanager(self) -> None:
        self.datamanager = Data_manager(
            iD_datasets=[self.iD],
            OoD_datasets=self.OoD,
            labelled_size=self.labelled_size,
            pool_size=self.pool_size,
            OoD_ratio=self.OOD_ratio,
            test_iD_size=None,
        )
        print("initialised datamanager")

    # overrides set_sampler
    def set_sampler(self, sampler) -> None:
        if sampler == "random":
            from .helpers.sampler import random_sample

            self.sampler = random_sample
        elif sampler == "highest-entropy":
            from .helpers.sampler import uncertainity_sampling_highest_entropy

            self.sampler = uncertainity_sampling_highest_entropy
        elif sampler == "least-confidence":
            from .helpers.sampler import uncertainity_sampling_least_confident

            self.sampler = uncertainity_sampling_least_confident
        elif sampler == "extra_class_entropy":
            from .helpers.sampler import extra_class_sampler

            self.sampler = extra_class_sampler(self.extra_class_thresholding)

    # overrides set_model
    def set_model(self, model_name) -> None:
        if model_name == "base":
            self.model = get_model(
                model_name, num_classes=self.num_classes
            )  # needs rewrite maybe
        else:
            raise NotImplementedError
        self.model.to(self.device)

    # overrides create_plots
    def create_plots(self) -> None:
        tsne_plot = get_tsne_plot(self.data_manager, self.iD, self.model, self.device)
        self.writer.add_figure(
            tag=f"{self.metric}/{self.iD}/{self.experiment_settings.get('oracles', 'oracle')}/tsne",
            figure=tsne_plot,
        )

    def pool_predictions(self, pool_loader) -> Union[np.ndarray, np.ndarray]:
        yhat = []
        labels_list = []
        for (data, labels) in pool_loader:
            pred = self.model(data.to(self.device).float(), apply_softmax=True)
            yhat.append(pred.to("cpu").detach().numpy())
            labels_list.append(labels)
        predictions = np.concatenate(yhat)
        labels_list = np.concatenate(labels_list)
        return predictions, labels_list

    def create_dataloader(self) -> None:
        result_tup = create_dataloader(
            self.datamanager, self.batch_size, 0.1, validation_source="train"
        )
        self.train_loader = result_tup[0]
        self.test_loader = result_tup[1]
        self.pool_loader = result_tup[2]
        self.val_loader = result_tup[3]

    def create_optimizer(self) -> None:
        self.optimizer = optim.SGD(
            self.model.parameters(),
            weight_decay=self.weight_decay,
            lr=self.lr,
            momentum=self.momentum,
            nesterov=self.nesterov,
        )

    def create_criterion(self) -> None:
        self.criterion = nn.CrossEntropyLoss()

    # overrides load_settings
    def load_settings(self) -> None:
        # active Learning settings
        self.oracle_stepsize = self.current_experiment.get("oracle_stepsize", 100)
        self.oracle_steps = self.current_experiment.get("oracle_steps", 10)
        self.iD = self.current_experiment.get("iD", "Cifar10")
        self.OoD = self.current_experiment.get("OoD", ["Fashion_MNIST"])
        self.labelled_size = self.current_experiment.get("labelled_size", 3000)
        self.pool_size = self.current_experiment.get("pool_size", 20000)
        self.OOD_ratio = self.current_experiment.get("OOD_ratio", 0.0)

        # training settings
        self.epochs = self.current_experiment.get("epochs", 100)
        self.batch_size = self.current_experiment.get("batch_size", 128)
        self.weight_decay = self.current_experiment.get("weight_decay", 1e-4)
        self.metric = self.current_experiment.get("metric", 10)
        self.lr = self.current_experiment.get("lr", 0.1)
        self.nesterov = self.current_experiment.get("nesterov", False)
        self.momentum = self.current_experiment.get("momentum", 0.9)
        self.lr_sheduler = self.current_experiment.get("lr_sheduler", True)
        self.num_classes = self.current_experiment.get("num_classes", 10)
        self.validation_split = self.current_experiment.get("validation_split", "train")
        self.validation_source = self.current_experiment.get("validation_source", 0.3)
        # self.criterion = self.current_experiment.get("criterion", "crossentropy")
        self.create_criterion()
        self.metric = self.current_experiment.get("metric", "accuracy")
        # logging
        self.verbose = self.current_experiment.get("verbose", 1)

        # _create_log_path_al(self.OOD_ratio)

        # extra class
        if self.current_experiment["exp_type"] == "extra_class":
            self.extra_class_thresholding = self.current_experiment.get(
                "extra_class_thresholding", 0.1
            )

            self.num_classes += 1
            # self.set_model(
            #     self.current_experiment.get("model", "base"), self.num_classes
            # )
        self.oracle = self.current_experiment.get("oracle", "highest-entropy")
        self.set_sampler(self.oracle)
        self.exp_name = self.current_experiment.get("exp_name", "standard_name")

    # overrides perform_experiment
    def perform_experiment(self):

        self.train_loss_hist = []
        check_path = os.path.join(
            self.log_path, "status_manager_dir", "intial_statusmanager.csv"
        )
        if os.path.exists(check_path):
            self.datamanager.status_manager = pd.read_csv(check_path, index_col=0)
            self.datamanager.reset_pool()
            print("loaded statusmanager from file")
        else:
            # self.datamanager.reset_pool()
            save_path = os.path.join(self.log_path, "status_manager_dir")
            self.datamanager.create_merged_data(path=save_path)

            print("created new statusmanager")
        self.current_oracle_step = 0
        for oracle_s in range(self.oracle_steps):
            self.set_model(
                self.current_experiment.get("model", "base"),
            )
            self.create_dataloader()
            self.create_optimizer()

            # , train_loader, val_loader, optimizer, criterion, device
            self.train(
                self.train_loader,
                self.val_loader,
                self.optimizer,
                self.criterion,
                self.device,
            )
            self.test()

            self.current_oracle_step += 1
            if len(self.pool_loader) > 0:
                (
                    pool_predictions,
                    pool_labels_list,
                ) = self.pool_predictions(self.pool_loader)

                self.sampler(
                    self.datamanager,
                    number_samples=self.oracle_stepsize,
                    net=self.model,
                    predictions=pool_predictions,
                )

                test_predictions, test_labels = self.pool_predictions(self.test_loader)

                test_accuracy = accuracy(test_labels, test_predictions)
                f1_score = f1(test_labels, test_predictions)

                dict_to_add = {
                    "test_loss": self.avg_test_loss,
                    "train_loss": self.avg_train_loss_hist,
                    "test_accuracy": test_accuracy,
                    "train_accuracy": self.avg_train_acc_hist,
                    "f1": f1_score,
                }

                print(dict_to_add)
                # if self.metric.lower() == "auroc":
                #     auroc_score = auroc(self.data_manager, oracle_s)

                #     dict_to_add = {"auroc": auroc_score}

                self.datamanager.add_log(
                    writer=self.writer,
                    oracle=self.oracle,
                    dataset=self.iD,
                    metric=self.metric,
                    log_dict=dict_to_add,
                    ood_ratio=self.OOD_ratio,
                    exp_name=self.exp_name,
                )
                self.save_al_logs()
        self.datamanager.status_manager.to_csv(
            os.path.join(
                self.log_path,
                "status_manager_dir",
                f"{self.exp_name}-result-statusmanager.csv",
            )
        )
