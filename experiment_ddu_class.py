from typing import Dict, List, Union

# python
import datetime
import os
import json
import numpy as np
from numpy.random import sample
from scipy.sparse import construct
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
from .model.model_files.small_resmodel_original import resmodel20
from .helpers.early_stopping import EarlyStopping
from .helpers.plots import get_tsne_plot
from .helpers.sampler import DDU_sampler
from .data.datamanager import Data_manager


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
        super().__init__(basic_settings, log_path)
        self.basic_settings = basic_settings
        self.exp_settings = exp_settings
        self.log_path = log_path
        self.writer = writer
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        if self.device == "cuda":
            torch.backends.cudnn.benchmark = True

    # overrides train
    def train(
        self, train_dataloader, val_dataloader, optimizer, criterion, device, **kwargs
    ):
        """train [main training function of the project]

        [extended_summary]

        Args:
            train_dataloader ([torch.Dataloader]): [dataloader with the training data]
            optimizer ([torch.optim]): [optimizer for the modelwork]
            criterion ([Loss function]): [Pytorch loss function]
            device ([str]): [device to train on cpu/cuda]
            epochs (int, optional): [epochs to run]. Defaults to 5.
            **kwargs (verbose and validation dataloader)
        Returns:
            [tupel(trained modelwork, train_loss )]:
        """
        # verbose = kwargs.get("verbose", 1)

        if self.verbose > 0:
            print("\nTraining with device :", device)
            print("Number of Training Samples : ", len(train_dataloader.dataset))
            if val_dataloader is not None:
                print("Number of Validation Samples : ", len(val_dataloader.dataset))
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
            for batch_idx, (data, target) in enumerate(train_dataloader):
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

            avg_train_loss = train_loss / len(train_dataloader)
            avg_train_acc = train_acc / len(train_dataloader.dataset)

            if epoch % 1 == 0:
                if validation:
                    val_loss = 0
                    val_acc = 0
                    self.model.eval()  # prep self.model for evaluation
                    with torch.no_grad():
                        for vdata, vtarget in val_dataloader:
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

                    avg_val_loss = val_loss / len(self.val_dataloader)
                    avg_val_acc = val_acc / len(self.val_dataloader.dataset)

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
        self.model.eval()
        for (t_data, t_target) in self.test_dataloader:
            t_data, t_target = (
                t_data.to(self.device).float(),
                t_target.to(self.device).long(),
            )

            yhat = self.model(t_data)
            yhat.to(self.device).long()
            t_loss = self.criterion(yhat, t_target)
            test_loss += t_loss

        self.avg_test_loss = test_loss.to("cpu").detach().numpy() / len(
            self.test_dataloader
        )  # return avg testloss

    # overrides save_settings
    def save_settings(self) -> None:
        log_config_path = os.path.join(self.log_dir + ".json")
        with open(log_config_path, "w") as f:
            json.dump(self.current_experiment, f)

    def save_al_logs(self) -> None:
        log_df = self.statusmanager.get_logs()
        with open(self.log_path, mode="w", encoding="utf-8") as logfile:
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
        self.datamanager = Data_manager(self.iD)

        pass

    def get_embeddings(
        self,
        model,
        loader: torch.utils.data.DataLoader,
        num_dim: int,
        dtype,
        device,
        storage_device,
    ):
        num_samples = len(loader.dataset)
        embeddings = torch.empty(
            (num_samples, num_dim), dtype=dtype, device=storage_device
        )
        labels = torch.empty(num_samples, dtype=torch.int, device=storage_device)

        with torch.no_grad():
            start = 0
            for data, label in tqdm(loader):
                data = data.to(device)
                label = label.to(device)

                if isinstance(model, nn.DataParallel):
                    out = model.module(data)
                    out = model.module.feature
                else:
                    out = model(data)
                    out = model.feature

                end = start + len(data)
                embeddings[start:end].copy_(out, non_blocking=True)
                labels[start:end].copy_(label, non_blocking=True)
                start = end

        return embeddings, labels

    def gmm_forward(self, model, gaussians_model, data_B_X):

        if isinstance(model, nn.DataParallel):
            features_B_Z = model.module(data_B_X)
            features_B_Z = model.module.feature
        else:
            features_B_Z = model(data_B_X)
            features_B_Z = model.feature

        log_probs_B_Y = gaussians_model.log_prob(features_B_Z[:, None, :])

        return log_probs_B_Y

    def gmm_evaluate(
        self, model, gaussians_model, loader, device, num_classes, storage_device
    ):

        num_samples = len(loader.dataset)
        logits_N_C = torch.empty(
            (num_samples, num_classes), dtype=torch.float, device=storage_device
        ).cuda()
        labels_N = torch.empty(
            num_samples, dtype=torch.int, device=storage_device
        ).cuda()

        with torch.no_grad():
            start = 0
            for data, label in tqdm(loader):
                data = data.to(device)
                label = label.to(device)

                logit_B_C = gmm_forward(model, gaussians_model, data)

                end = start + len(data)
                logits_N_C[start:end].copy_(logit_B_C, non_blocking=True)
                labels_N[start:end].copy_(label, non_blocking=True)
                start = end

        return logits_N_C, labels_N

    def gmm_get_logits(self, gmm, embeddings):

        log_probs_B_Y = gmm.log_prob(embeddings[:, None, :])
        return log_probs_B_Y

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
                except RuntimeError as e:
                    if "cholesky" in str(e):
                        continue
                except ValueError as e:
                    if "The parameter covariance_matrix has invalid values" in str(e):
                        continue
                break

        return gmm, jitter_eps

    # overrides set_sampler
    def set_sampler(self, sampler) -> None:
        self.sampler = DDU_sampler

    # overrides set_writer
    def set_writer(self, log_path) -> None:
        pass

    # overrides set_model
    def set_model(self, model_name) -> None:
        if model_name == "base_small_resmodel":
            self.model = resmodel20(
                num_classes=self.num_classes, similarity=self.similarity
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

    # overrides save_logs
    def save_logs(self) -> None:
        pass

    def pool_predictions(self, pool_dataloader) -> Union[np.ndarray, np.ndarray]:
        yhat = []
        labels_list = []
        for (data, labels) in pool_dataloader:
            pred = self.model(
                data.to(self.device).float(), get_test_model=True, apply_softmax=True
            )
            yhat.append(pred.to("cpu").detach().numpy())
            labels_list.append(labels)
        predictions = np.concatenate(yhat)
        labels_list = np.concatenate(labels_list)
        return predictions, labels_list

    #!TODO
    @torch.no_grad()
    def create_dataloader(self) -> None:
        pass

    def create_optimizer(self) -> None:
        self.optimizer = optim.SGD(
            self.model.parameters,
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
        self.criterion = self.current_experiment.get("criterion", "crossentropy")
        self.metric = self.current_experiment.get("metric", "accuracy")
        # logging
        self.verbose = self.current_experiment.get("verbose", 1)

        _create_log_path_al(self.OOD_ratio)

        self.set_model(self.current_experiment.get("model", "base_small_resmodel"))
        self.set_writer(self.log_path)
        self.set_sampler(self.current_experiment.get("oracles", "highest-entropy"))

    # overrides perform_experiment
    def perform_experiment(self):
        self.datamanager = None

        for experiment in self.experiment_settings:
            self.current_experiment = experiment
            self.train_loss_hist = []
            self.load_settings()
            self.construct_datamanager()
            self.current_oracle_step = 0
            for oracle_s in self.oracle_steps:

                self.create_dataloader()
                self.create_optimizer()

                self.train(self.dataloader, self.optimizer, self.criterion, self.device)
                self.test()
                self.model.eval()
                embeddings, labels = self.get_embeddings(
                    self.model,
                    self.train_dataloader,
                    num_dim=512,
                    dtype=torch.double,
                    device=self.device,
                    storage_device=self.device,
                )
                print("fitting gmm")
                gaussians_model, jitter_eps = self.gmm_fit(
                    embeddings=embeddings, labels=labels, num_classes=10
                )
                self.gaussians_model = gaussians_model
                if len(self.pool_dataloader) > 0:

                    pool_predictions, pool_labels_list = self.get_pool_predictions(
                        self.model,
                        self.pool_dataloader,
                        device=self.device,
                        return_labels=True,
                    )
                    # class_prob = class_probs(train_loader)
                    pool_predictions = torch.from_numpy(pool_predictions).cuda()
                    print("finished pool prediction")
                    logits, labels = self.gmm_evaluate(
                        self.model,
                        gaussians_model,
                        self.pool_dataloader,
                        device=self.device,
                        num_classes=10,
                        storage_device=self.device,
                    )

                    # logits, labels = (
                    #     logits.detach().to("cpu").numpy(),
                    #     labels.detach().to("cpu").numpy(),
                    # )
                    print("finished gmm evaluation")
                    # samples from unlabelled pool predictions
                    self.sampler(
                        dataset_manager=self.data_manager,
                        number_samples=self.oracle_stepsize,
                        net=self.moode,
                        predictions=pool_predictions,
                        gmm_logits=logits,
                        class_probs=pool_predictions,
                    )

                    test_predictions, test_labels, _ = self.get_pool_predictions(
                        self.test_loader
                    )
                    train_predictions, train_labels, _ = self.get_pool_predictions(
                        self.train_dataloader
                    )

                    if self.metric.lower() == "accuracy":
                        test_accuracy = accuracy(test_labels, test_predictions)
                        train_accuracy = accuracy(train_labels, train_predictions)

                        dict_to_add = {
                            "test_loss": self.avg_test_loss,
                            "train_loss": self.avg_train_loss,
                            "test_accuracy": test_accuracy,
                            "train_accuracy": train_accuracy,
                        }
                        print(dict_to_add)

                    elif self.metric.lower() == "f1":
                        f1_score = f1(test_labels, test_predictions)
                        dict_to_add = {"f1": f1_score}
                    elif self.metric.lower() == "auroc":
                        auroc_score = auroc(self.data_manager, oracle_s)

                        dict_to_add = {"auroc": auroc_score}

                    # TODO wait for final version
                    # self.data_manager.add_log(
                    #     writer=self.writer,
                    #     oracle=self.oracle,
                    #     dataset=self.iD,
                    #     metric=self.metric,
                    #     log_dict=dict_to_add,
                    #     param_dict=param_dict,
                    # )

                    # self.save_logs(data_manager, log_path)
                self.current_oracle_step += 1