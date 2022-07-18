import logging
from typing import Dict

import numpy as np
from joblib import dump


class Trainer:
    def __init__(self, datasets: Dict[str, np.ndarray], model, training_config):
        self.model = model
        self.cfg = training_config
        self.datasets = datasets
        self.logger = logging.getLogger(__name__)

    def __call__(self):
        X, y = self.datasets["train"]["features"], self.datasets["train"]["labels"]
        if self.cfg.USE_CV:
            pass
            # cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
            # # evaluate the model and collect the scores
            # n_scores = cross_val_score(self.model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
        else:
            self.model.fit(X, y)
