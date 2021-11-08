import gc
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


# project
from .experiment_base import experiment_base
from .model.model_files.small_resnet_original import resnet20
from .helpers.early_stopping import EarlyStopping
from .helpers.plots import get_tsne_plot, density_plot
from .data.datahandler_for_array import create_dataloader
from .data.datamanager import Data_manager
from .helpers.measures import accuracy, auroc, f1


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


def fgsm_attack(image, epsilon, data_grad):
    # Collect the element-wise sign of the data gradient
    sign_data_grad = data_grad.sign()
    # Create the perturbed image by adjusting each pixel of the input image
    perturbed_image = image + epsilon * sign_data_grad
    # Adding clipping to maintain [0,1] range
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    # Return the perturbed image
    return perturbed_image


class experiment_gen_odin(experiment_base):
    def __init__(
        self,
        basic_settings: Dict,
        exp_settings: Dict,
        log_path: str,
        writer: SummaryWriter,
    ) -> None:
        super().__init__(basic_settings, log_path)
        self.log_path = log_path
        self.writer = writer
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.current_experiment = basic_settings | exp_settings

    # overrides train
    def train(
        self, train_dataloader, val_dataloader, optimizer, criterion, device, **kwargs
    ):
        """train [main training function of the project]

        [extended_summary]

        Args:
            train_dataloader ([torch.Dataloader]): [dataloader with the training data]
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
        elif sampler == "highest entropy":
            from .helpers.sampler import uncertainity_sampling_highest_entropy

            self.sampler = uncertainity_sampling_highest_entropy
        elif sampler == "least confidence":
            from .helpers.sampler import uncertainity_sampling_least_confident

            self.sampler = uncertainity_sampling_least_confident
        else:
            raise NotImplementedError

    # overrides set_model
    def set_model(self, model_name) -> None:
        if model_name == "base_small_resnet":
            self.model = resnet20(
                num_classes=self.num_classes, similarity=self.similarity
            )  # needs rewrite maybe
        else:
            raise NotImplementedError
        self.model.to(self.device)

    # overrides create_plots
    def create_plots(self, plot_name, pret_preds, gs, hs, targets, oracle_step) -> None:
        if plot_name == "tsne":
            tsne_plot = get_tsne_plot(
                self.data_manager, self.iD, self.model, self.device
            )
            self.writer.add_figure(
                tag=f"{self.metric}/{self.iD}/{self.experiment_settings.get('oracles', 'oracle')}/tsne",
                figure=tsne_plot,
            )
        elif plot_name == "density":
            density_plot(pret_preds, gs, hs, targets, self.writer, oracle_step)
        else:
            raise NotImplementedError  # for layer analsis i guess -> maybe split into seperate functions

    # overrides save_logs
    def save_logs(self) -> None:
        pass

    def pool_predictions(
        self, pool_loader
    ) -> Union[np.ndarray, np.ndarray, np.ndarray]:
        yhat = []
        labels_list = []
        weighting_factor_list = []
        for (data, labels) in pool_loader:
            if self.bugged_and_working:
                tuple_data = self.model(
                    data.to(self.device).float(),
                    get_test_model=True,
                    apply_softmax=False,
                )
            else:
                tuple_data = self.model(
                    data.to(self.device).float(),
                    get_test_model=True,
                    apply_softmax=True,
                )

            pred = tuple_data[0]
            weighting_factor = tuple_data[1]
            weighting_factor_list.append(weighting_factor.to("cpu").detach().numpy())

            yhat.append(pred.to("cpu").detach().numpy())
            labels_list.append(labels)
        predictions = np.concatenate(yhat)
        labels_list = np.concatenate(labels_list)
        weighting_factor_list = np.concatenate(weighting_factor_list)
        return predictions, labels_list, weighting_factor_list

    def create_dataloader(self) -> None:
        result_tup = create_dataloader(
            self.datamanager, self.batch_size, 0.1, validation_source="train"
        )
        self.train_loader = result_tup[0]
        self.test_loader = result_tup[1]
        self.pool_loader = result_tup[2]
        self.val_loader = result_tup[3]

    def create_optimizer(self) -> None:
        base_params = []
        gen_odin_params = []
        for name, param in self.model.named_parameters():
            if name not in [
                "h_func.bias",
                "h_func.weights",
                "scaling_factor",
            ]:
                base_params.append(param)  # can't do the name tupel
            else:
                if self.verbose >= 2:
                    print("added name: ", name)
                gen_odin_params.append(param)

        self.optimizer = optim.SGD(
            [
                {"params": base_params},
                {"params": gen_odin_params, "weight_decay": 0.0},
            ],
            weight_decay=self.weight_decay,
            lr=self.lr,
            momentum=self.momentum,
            nesterov=self.nesterov,
        )

    def create_criterion(self) -> None:
        self.criterion = nn.CrossEntropyLoss()

    def pertube_image(self, pool_loader, val_loader):
        gs = []
        hs = []
        pert_preds = []
        targets = []
        device = self.device

        epsi_list = [0.0025, 0.005, 0.01, 0.02, 0.04, 0.08]
        best_eps = 0
        scores = []
        self.model.eval()
        for eps in tqdm(epsi_list):
            preds = 0
            for batch_idx, (data, target) in enumerate(val_loader):
                self.model.zero_grad(set_to_none=True)
                backward_tensor = torch.ones((data.size(0), 1)).float().to(device)
                data, target = data.to(device).float(), target.to(device).long()
                data.requires_grad = True
                output, g, h = self.model(data, get_test_model=True, apply_softmax=True)
                pred, _ = output.max(dim=-1, keepdim=True)

                pred.backward(backward_tensor)
                pert_imgage = fgsm_attack(data, epsilon=eps, data_grad=data.grad.data)
                del data, output, target, g, h
                gc.collect()

                yhat = self.model(pert_imgage, apply_softmax=True)
                pred = torch.max(yhat, dim=-1, keepdim=False, out=None).values
                preds += torch.sum(pred)
                del pred, yhat, pert_imgage
                gc.collect()
            scores.append(preds.detach().cpu().numpy())

        torch.cuda.empty_cache()
        self.model.zero_grad(set_to_none=True)
        eps = epsi_list[np.argmax(scores)]
        del scores

        targets = []
        for batch_idx, (data, target) in enumerate(pool_loader):
            self.model.zero_grad(set_to_none=True)
            backward_tensor = torch.ones((data.size(0), 1)).float().to(device)
            data, target = data.to(device).float(), target.to(device).long()
            data.requires_grad = True
            output, g, h = self.model(data, get_test_model=True, apply_softmax=True)
            pred, _ = output.max(dim=-1, keepdim=True)

            pred.backward(backward_tensor)
            pert_imgage = fgsm_attack(data, epsilon=eps, data_grad=data.grad.data)
            targets.append(target.to("cpu").numpy().astype(np.float16))
            del data, output, target, g, h

            with torch.no_grad():
                pert_pred, g, h = self.model(
                    pert_imgage, get_test_model=True, apply_softmax=True
                )
                gs.append(g.detach().to("cpu").numpy().astype(np.float16))
                hs.append(h.detach().to("cpu").numpy().astype(np.float16))
                pert_preds.append(pert_pred.detach().to("cpu").numpy())
            del pert_pred, g, h
            gc.collect()

        return pert_preds, gs, hs, targets

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

        self.plotsettings = self.current_experiment.get(
            "plotsettings", {"do_plot": True, "density_plot": True, "layer_plot": False}
        )

        _create_log_path_al(self.OOD_ratio)
        self.similarity = self.current_experiment.get("similarity", "E")
        self.set_model(
            self.current_experiment.get("model", "base_small_resnet"), self.similarity
        )
        self.set_writer(self.log_path)
        self.set_sampler(self.current_experiment.get("oracles", "highest-entropy"))
        self.do_pertubed_images = self.current_experiment.get(
            "do_pertubed_images", False
        )
        self.do_desity_plot = self.current_experiment.get("do_desity_plot", False)
        self.bugged_and_working = self.current_experiment.get(
            "bugged_and_working", None
        )
        print("loaded settings")
        if self.current_experiment.get("bugged_and_working", None) is None:
            bugged_and_working = self.current_experiment.get("bugged_and_working", True)
            print(
                f"INFO ---- flag bugged_and_working is not set. Using default value of {bugged_and_working}"
            )
        else:
            bugged_and_working = self.current_experiment["bugged_and_working"]
            print(f"INFO ---- flag bugged_and_working is set to {bugged_and_working}")

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

                self.current_oracle_step += 1
                if len(self.pool_loader) > 0:
                    (
                        pool_predictions,
                        pool_labels_list,
                        weighting_factor,
                    ) = self.get_pool_predictions(self.pool_loader)

                    self.sampler(self.datamanager, number_samples=self.oracle_stepsize)

                    test_predictions, test_labels, _ = self.get_pool_predictions(
                        self.test_loader
                    )
                    train_predictions, train_labels, _ = self.get_pool_predictions(
                        self.train_loader
                    )

                    if self.do_desity_plot:
                        pert_preds, gs, hs, targets = self.pertube_image(
                            self.pool_loader,
                            self.val_loader,
                        )

                        self.create_plots(
                            "density", pert_preds, gs, hs, targets, oracle_s
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
