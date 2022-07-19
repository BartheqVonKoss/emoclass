"""Holds dictionaries that consists of parameters to be searched during optimization."""
import optuna

lr_param_distributions = {"C": optuna.distributions.LogUniformDistribution(1e-10, 1e10),
                          "fit_intercept": optuna.distributions.CategoricalDistribution([True, False]),
                          "warm_start": optuna.distributions.CategoricalDistribution([True, False])}

rf_param_distributions = {"n_estimators": optuna.distributions.IntUniformDistribution(50, 250),
                          "max_depth": optuna.distributions.IntUniformDistribution(3, 10),
                          "ccp_alpha": optuna.distributions.LogUniformDistribution(1e-6, 1e-1),
                          "oob_score": optuna.distributions.CategoricalDistribution([True, False]),
                          "max_features": optuna.distributions.CategoricalDistribution(["sqrt", "log2", None]),
                          "criterion": optuna.distributions.CategoricalDistribution(["gini", "entropy", "log_loss"])}
