import enum
from typing import Dict, Union, List, NoReturn
from abc import ABC, abstractmethod
import torch


class experiment_base(ABC):
    """experiment_base [abstract base class other experiments ]"""

    def __init__(self, experiment_settings: List[Dict], log_path: str) -> NoReturn:
        self.experiment_settings = experiment_settings
        self.log_path = log_path
        self.devic = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        pass

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
    def set_writer(self) -> NoReturn:
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
