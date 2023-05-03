import logging
import os

from guido.models.text_classifier import sentence_bert
import hydra
import mlflow
from omegaconf import DictConfig, ListConfig

logger = logging.getLogger(__name__)


def log_params_from_omegaconf_dict(params):
    for param_name, element in params.items():
        _explore_recursive(param_name, element)


def _explore_recursive(parent_name, element):
    if isinstance(element, DictConfig):
        for k, v in element.items():
            if isinstance(v, DictConfig) or isinstance(v, ListConfig):
                _explore_recursive(f'{parent_name}.{k}', v)
            else:
                mlflow.log_param(f'{parent_name}.{k}', v)
    elif isinstance(element, ListConfig):
        for i, v in enumerate(element):
            mlflow.log_param(f'{parent_name}.{i}', v)


@hydra.main(version_base=None, config_path='config/', config_name='config')
def train_classifier(conf: DictConfig):
    """

    :param conf: hydra config
    :return: None
    """
    mlflow.set_tracking_uri('file://' + os.getcwd() + '/mlruns')
    mlflow.set_experiment(conf.mlflow.experiment_name)

    with mlflow.start_run():
        log_params_from_omegaconf_dict(conf)
        sentence_bert.run(conf)


if __name__ == "__main__":
    train_classifier()
