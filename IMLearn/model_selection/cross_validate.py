from __future__ import annotations
from copy import deepcopy
from typing import Tuple, Callable
import numpy as np
import pandas as pd

from IMLearn import BaseEstimator


def cross_validate(estimator: BaseEstimator, X: np.ndarray, y: np.ndarray,
                   scoring: Callable[[np.ndarray, np.ndarray, ...], float], cv: int = 5) -> Tuple[float, float]:
    """
    Evaluate metric by cross-validation for given estimator

    Parameters
    ----------
    estimator: BaseEstimator
        Initialized estimator to use for fitting the data

    X: ndarray of shape (n_samples, n_features)
       Input data to fit

    y: ndarray of shape (n_samples, )
       Responses of input data to fit to

    scoring: Callable[[np.ndarray, np.ndarray, ...], float]
        Callable to use for evaluating the performance of the cross-validated model.
        When called, the scoring function receives the true- and predicted values for each sample
        and potentially additional arguments. The function returns the score for given input.

    cv: int
        Specify the number of folds.

    Returns
    -------
    train_score: float
        Average train score over folds

    validation_score: float
        Average validation score over folds
    """

    sub_sets_X = np.array_split(X, cv)
    sub_sets_y = np.array_split(y, cv)

    training_errors = np.zeros(cv)
    validation_errors = np.zeros(cv)

    for i in range(cv):
        training_X = np.concatenate(sub_sets_X[:i] + sub_sets_X[i+1:])
        training_y = np.concatenate(sub_sets_y[:i] + sub_sets_y[i+1:])
        val_set_X = sub_sets_X[i]
        val_set_y = sub_sets_y[i]

        estimator.fit(training_X, training_y)
        training_errors[i] = scoring(estimator.predict(training_X), training_y)
        validation_errors[i] = scoring(estimator.predict(val_set_X), val_set_y)

    return float(np.mean(training_errors)), float(np.mean(validation_errors))