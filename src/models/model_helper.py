"""Holds a dictionary that consists of pairs of name - model."""
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from config.training_config import get_training_config

train_cfg = get_training_config()

model_helper = {
    "logistic_regression": LogisticRegression(
        multi_class='multinomial',
        solver='lbfgs',
        max_iter=train_cfg.MAX_ITER,
        penalty="l2",
        C=1e-4,
        n_jobs=train_cfg.JOBS_NO,
        verbose=2),
    "random_forest": RandomForestClassifier(
        verbose=2,
        n_jobs=train_cfg.JOBS_NO),
}
