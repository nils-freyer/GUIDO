from dataclasses import dataclass
from hydra.core.config_store import ConfigStore


@dataclass
class Mlflow:
    experiment_name: str
    url: str


@dataclass
class Optimizer:
    dropout: float
    lr: float


@dataclass
class Training:
    batch_size: int
    epochs: int


@dataclass
class AppConfig:
    optimizer: Optimizer
    training: Training
    mlflow: Mlflow


cs = ConfigStore.instance()
# Registering the Config class with the name 'config'.
cs.store(name="config", node=AppConfig)
