import enum
from typing import Dict, Union, List, NoReturn
from abc import ABC, abstractmethod
import torch
from torch.utils.tensorboard.writer import SummaryWriter


class experiment_base(ABC):
    """experiment_base [abstract base class other experiments ]"""

    def __init__(
        self,
        basic_settings: Dict,
        exp_settings: Dict,
        log_path: str,
        writer: SummaryWriter,
    ) -> NoReturn:
        self.log_path = log_path
        self.writer = writer
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.current_experiment = basic_settings | exp_settings

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def test(self):
        pass

    @abstractmethod
    def load_settings(self) -> NoReturn:
        pass

    @abstractmethod
    def save_settings(self) -> NoReturn:
        pass

    @abstractmethod
    def construct_datamanager(self) -> NoReturn:
        pass

    @abstractmethod
    def set_sampler(self) -> NoReturn:
        pass

    @abstractmethod
    def set_model(self) -> NoReturn:
        pass

    @abstractmethod
    def create_plots(self) -> NoReturn:
        pass

    @abstractmethod
    def save_logs(self) -> NoReturn:
        pass

    @abstractmethod
    def perform_experiment(self):
        self.datamanager = None
        pass
