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

# from .model.model_files.small_resnet.model_original import resmodel20 #
from .model.get_model import get_model
from .helpers.early_stopping import EarlyStopping
from .helpers.plots import get_tsne_plot
from .helpers.sampler import DDU_sampler
from .data.data_manager import Data_manager
from .data.datahandler_for_array import create_dataloader
from .helpers.measures import accuracy, auroc, f1

from .data.deprecated.datahandler_for_array import (
    create_dataloader as create_dataloader_old,
)
from .data.deprecated.data_manager import get_data_manager as get_data_manager_old
from .data.deprecated.sampler import DDU_sampler as DDU_sampler_old


def verbosity(message, verbose, epoch):
    if verbose == 1:
        if epoch % 10 == 0:
            print(message)
    elif verbose == 2:
        print(message)
    return None


def centered_cov_torch(x):
    n = x.shape[0]
    res = 1 / (n - 1) * x.t().mm(x)
    return res


def compute_density(logits, class_probs):
    return torch.sum((torch.exp(logits) * class_probs), dim=1)


DOUBLE_INFO = torch.finfo(torch.double)
JITTERS = [0, DOUBLE_INFO.tiny] + [10 ** exp for exp in range(-308, 0, 1)]


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


class experiment_ddu(experiment_base):
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

        self.construct_data_manager()

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
        log_df = self.data_manager.get_logs()
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

    # overrides construct_data_manager
    def construct_data_manager(self) -> None:
        self.data_manager = Data_manager(
            iD_datasets=[self.iD],
            OoD_datasets=self.OoD,
            labelled_size=self.labelled_size,
            pool_size=self.pool_size,
            OoD_ratio=self.OOD_ratio,
            test_iD_size=None,
            subclass=self.current_experiment.get("subclass", {"do_subclass": False}),
        )
        print("initialised data_manager")

    def get_embeddings(
        self,
        num_dim: int,
        dtype,
    ):
        num_samples = len(self.train_loader.dataset)
        embeddings = []
        labels = []

        with torch.no_grad():
            start = 0
            for data, label in tqdm(self.train_loader):
                data = data.to(self.device)
                label = label.to(self.device)

                if isinstance(self.model, nn.DataParallel):
                    out = self.model.module(data)
                    out = self.model.module.feature
                else:
                    out = self.model(data)
                    out = self.model.feature

                embeddings.append(out)
                labels.append(label)

        embeddings = torch.cat(embeddings, axis=0)
        labels = torch.cat(labels, axis=0)
        self.embeddings = embeddings
        self.labels = labels
        return embeddings, labels

    def gmm_forward(self, data_B_X):

        if isinstance(self.model, nn.DataParallel):
            features_B_Z = self.model.module(data_B_X)
            features_B_Z = self.model.module.feature
        else:
            features_B_Z = self.model(data_B_X)
            features_B_Z = self.model.feature

        log_probs_B_Y = self.gaussians_model.log_prob(features_B_Z[:, None, :])

        return log_probs_B_Y

    def gmm_evaluate(
        self,
    ):

        num_samples = len(self.pool_loader.dataset)
        logits_N_C = torch.empty(
            (num_samples, self.num_classes), dtype=torch.float, device=self.device
        ).cuda()
        labels_N = torch.empty(num_samples, dtype=torch.int, device=self.device).cuda()

        with torch.no_grad():
            start = 0
            for data, label in tqdm(self.pool_loader):
                data = data.to(self.device)
                label = label.to(self.device)

                logit_B_C = self.gmm_forward(data)

                end = start + len(data)

                logits_N_C[start:end].copy_(logit_B_C, non_blocking=True)
                labels_N[start:end].copy_(label, non_blocking=True)
                start = end

        return logits_N_C, labels_N

    def gmm_fit(self, embeddings, labels, num_classes):
        with torch.no_grad():
            classwise_mean_features = torch.stack(
                [torch.mean(embeddings[labels == c], dim=0) for c in range(num_classes)]
            )
            classwise_cov_features = torch.stack(
                [
                    centered_cov_torch(
                        embeddings[labels == c] - classwise_mean_features[c]
                    )
                    for c in range(num_classes)
                ]
            )
        gmm = None
        with torch.no_grad():
            for jitter_eps in JITTERS:
                try:
                    jitter = (
                        jitter_eps
                        * torch.eye(
                            classwise_cov_features.shape[1],
                            device=classwise_cov_features.device,
                        ).unsqueeze(0)
                    )
                    gmm = torch.distributions.MultivariateNormal(
                        loc=classwise_mean_features,
                        covariance_matrix=(classwise_cov_features + jitter),
                    )

                except:
                    continue
                break

        return gmm, jitter_eps

    # overrides set_sampler
    def set_sampler(self) -> None:
        self.sampler = DDU_sampler

    # overrides set_model
    def set_model(self, model_name) -> None:
        if model_name == "DDU":
            self.model = get_model(
                model_name,
                num_classes=self.num_classes,
                spectral_normalization=self.spectral_normalization,
                temp=self.temp,
            )  # needs rewrite maybe
        else:
            raise NotImplemented

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
            pred = self.model(data.to(self.device).float())
            yhat.append(pred.to("cpu").detach().numpy())
            labels_list.append(labels)
        predictions = np.concatenate(yhat)
        labels_list = np.concatenate(labels_list)
        return predictions, labels_list

    def create_dataloader(self) -> None:
        result_tup = create_dataloader(
            self.data_manager,
            self.batch_size,
            self.validation_split,
            validation_source=self.validation_source,
        )
        self.train_loader = result_tup[0]
        self.test_loader = result_tup[1]
        self.pool_loader = result_tup[2]
        self.val_loader = result_tup[3]

    def create_optimizer(self) -> None:
        self.optimizer = optim.Adam(
            self.model.parameters(), weight_decay=5e-4  # self.weight_decay
        )
        # self.optimizer = optim.SGD(
        #     self.model.parameters(),
        #     weight_decay=self.weight_decay,
        #     lr=self.lr,
        #     momentum=self.momentum,
        #     nesterov=self.nesterov,
        # )

    def create_criterion(self) -> None:
        self.criterion = nn.CrossEntropyLoss()

    # overrides load_settings
    def load_settings(self) -> None:
        self.exp_name = self.current_experiment.get("exp_name", "standard_name")
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
        self.validation_split = self.current_experiment.get("validation_split", 0.3)
        self.validation_source = self.current_experiment.get(
            "validation_source", "test"
        )
        # self.criterion = self.current_experiment.get("criterion", "crossentropy")
        self.create_criterion()
        self.metric = self.current_experiment.get("metric", "accuracy")
        # logging
        self.verbose = self.current_experiment.get("verbose", 1)
        self.spectral_normalization = self.current_experiment.get(
            "spectral_normalization", True
        )
        self.temp = self.current_experiment.get("temp", 1.0)
        self.set_sampler()
        self.oracle = self.current_experiment.get("oracle", "highest-entropy")

    def class_probs(self, dataloader):
        num_classes = self.num_classes
        class_n = len(dataloader.dataset)
        class_count = torch.zeros(num_classes)
        for data, label in dataloader:
            class_count += torch.Tensor(
                [torch.sum(label.to(self.device) == c) for c in range(num_classes)]
            )

        class_prob = class_count / class_n
        return class_prob

    # overrides perform_experiment
    def perform_experiment(self):
        self.train_loss_hist = []
        check_path = os.path.join(
            self.log_path, "status_manager_dir", "intial_statusmanager.csv"
        )
        if os.path.exists(check_path):
            self.data_manager.status_manager = pd.read_csv(check_path, index_col=0)
            self.data_manager.reset_pool()
            print("loaded statusmanager from file")
        else:
            # self.data_manager.reset_pool()
            save_path = os.path.join(self.log_path, "status_manager_dir")
            self.data_manager.create_merged_data(path=save_path)
            print("created new statusmanager")
        self.current_oracle_step = 0
        
        for oracle_s in range(self.oracle_steps):
            self.set_model("DDU") # hardcoded till we add larger models
            result_tup = self.create_dataloader()
            self.create_optimizer()

            self.train(
                self.train_loader,
                self.val_loader,
                self.optimizer,
                self.criterion,
                self.device,
            )
            self.test()

            self.model.eval()
            embeddings, labels = self.get_embeddings(num_dim=256, dtype=torch.double)

            print("fitting gmm")
            gaussians_model, jitter_eps = self.gmm_fit(
                embeddings=self.embeddings,
                labels=self.labels,
                num_classes=self.num_classes,
            )
            self.gaussians_model = gaussians_model
            if len(self.pool_loader) > 0:

                pool_predictions, pool_labels_list = self.pool_predictions(
                    self.pool_loader,
                )
                class_prob = self.class_probs(self.train_loader).to(self.device)
                pool_predictions = torch.from_numpy(pool_predictions).cuda()
                print("finished pool prediction")
                logits, labels = self.gmm_evaluate()
                print("finished gmm evaluation")

                densities = compute_density(logits, class_prob)
                densities = densities.detach().to("cpu").numpy()

                source_labels = self.data_manager.get_pool_source_labels()
                iD_Prob = 1 - np.exp(-densities)
                auroc_score = auroc(
                    iD_Prob,
                    source_labels,
                    self.writer,
                    self.current_oracle_step,
                    plot_auc=True,
                )

                # samples from unlabelled pool predictions
                self.sampler(
                    dataset_manager=self.data_manager,
                    number_samples=self.oracle_stepsize,
                    densities=densities,
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
                    "Pool_AUROC": auroc_score,
                }

                print(dict_to_add)
                # if self.metric.lower() == "auroc":
                #     auroc_score = auroc(self.data_manager, oracle_s)

                #     dict_to_add = {"auroc": auroc_score}

                self.data_manager.add_log(
                    writer=self.writer,
                    oracle=self.oracle,
                    dataset=self.iD,
                    metric=self.metric,
                    log_dict=dict_to_add,
                    ood_ratio=self.OOD_ratio,
                    exp_name=self.exp_name,
                )
                self.save_al_logs()
                # self.save_logs(self.data_manager, self.log_path)
        self.current_oracle_step += 1
        self.data_manager.status_manager.to_csv(
            os.path.join(
                self.log_path,
                "status_manager_dir",
                f"{self.exp_name}-result-statusmanager.csv",
            )
        )

    # def perform_old_experiment(self):
    #     self.data_manager = get_data_manager_old()
    #     self.sampler = DDU_sampler_old
    #     self.data_manager.create_merged_data(
    #         5000,
    #         pool_size=self.pool_size,
    #         labelled_size=self.labelled_size,
    #         OOD_ratio=self.OOD_ratio,
    #     )
    #     self.current_oracle_step = 0
    #     self.train_loss_hist = []

    #     for oracle_s in range(self.oracle_steps):

    #         result_tup = create_dataloader_old(
    #             self.data_manager,
    #             batch_size=self.batch_size,
    #             validation_source=self.validation_source,
    #             validation_split=0.1,
    #         )
    #         self.train_loader = result_tup[0]
    #         self.test_loader = result_tup[1]
    #         self.pool_loader = result_tup[2]
    #         self.val_loader = result_tup[3]

    #         self.create_optimizer()

    #         self.train(
    #             self.train_loader,
    #             self.val_loader,
    #             self.optimizer,
    #             self.criterion,
    #             self.device,
    #         )
    #         self.test()

    #         self.model.eval()
    #         self.embeddings, self.labels = self.get_embeddings(
    #             num_dim=256, dtype=torch.double
    #         )

    #         print("fitting gmm")
    #         gaussians_model, jitter_eps = self.gmm_fit(
    #             embeddings=self.embeddings,
    #             labels=self.labels,
    #             num_classes=self.num_classes,
    #         )
    #         self.gaussians_model = gaussians_model
    #         if len(self.pool_loader) > 0:

    #             pool_predictions, pool_labels_list = self.pool_predictions(
    #                 self.pool_loader,
    #             )
    #             # class_prob = class_probs(train_loader)
    #             pool_predictions = torch.from_numpy(pool_predictions).cuda()
    #             print("finished pool prediction")
    #             logits, labels = self.gmm_evaluate()

    #             # logits, labels = (
    #             #     logits.detach().to("cpu").numpy(),
    #             #     labels.detach().to("cpu").numpy(),
    #             # )
    #             print("finished gmm evaluation")
    #             # samples from unlabelled pool predictions
    #             self.sampler(
    #                 dataset_manager=self.data_manager,
    #                 number_samples=self.oracle_stepsize,
    #                 gmm_logits=logits,
    #                 class_probs=pool_predictions,
    #                 net=None,
    #                 predictions=None,
    #             )

    #             test_predictions, test_labels = self.pool_predictions(self.test_loader)

    #             test_accuracy = accuracy(test_labels, test_predictions)
    #             # f1_score = f1(test_labels, test_predictions)

    #             dict_to_add = {
    #                 "test_loss": self.avg_test_loss,
    #                 "train_loss": self.avg_train_loss_hist,
    #                 "test_accuracy": test_accuracy,
    #                 "train_accuracy": self.avg_train_acc_hist,
    #             }

    #             print(dict_to_add)
    #             # if self.metric.lower() == "auroc":
    #             #     auroc_score = auroc(self.data_manager, oracle_s)

    #             #     dict_to_add = {"auroc": auroc_score}

    #             self.data_manager.add_log(
    #                 writer=self.writer,
    #                 oracle=self.oracle,
    #                 dataset=self.iD,
    #                 metric=self.metric,
    #                 log_dict=dict_to_add,
    #             )
    #             self.save_al_logs()
    #             # self.save_logs(self.data_manager, self.log_path)
    #     self.current_oracle_step += 1
    #     self.data_manager.status_manager.to_csv(
    #         os.path.join(
    #             self.log_path,
    #             "status_manager_dir",
    #             f"{self.exp_name}-result-statusmanager.csv",
    #         )
    #     )
