import logging
from typing import Any, Dict

import numpy as np
import optuna
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from src.models.param_helper import (lr_param_distributions,
                                     rf_param_distributions)


class Trainer:
    def __init__(self, datasets: Dict[str, np.ndarray], model, training_config):
        self.model = model
        self.cfg = training_config
        self.datasets = datasets
        self.logger = logging.getLogger(__name__)
        self.param_distributions: Dict[str, Any] = None
        self.get_parameters_space()

    def __call__(self) -> None:
        X, y = self.datasets["train"]["features"], self.datasets["train"]["labels"]
        optuna_search = optuna.integration.OptunaSearchCV(self.model, self.param_distributions)
        # get best parameters after cross-validation and hyperparameters search
        optuna_search.fit(X, y)
        # set parameters to the model and train it
        self.model.set_params(**optuna_search.best_params_)
        self.model.fit(X, y)

    def get_parameters_space(self) -> None:
        """Method to get proper parameters space given the model."""
        if isinstance(self.model, LogisticRegression):
            self.param_distributions = lr_param_distributions
        if isinstance(self.model, RandomForestClassifier):
            self.param_distributions = rf_param_distributions
