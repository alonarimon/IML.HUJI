from __future__ import annotations
from typing import NoReturn

from numpy.linalg import pinv

from ...base import BaseEstimator
import numpy as np


class RidgeRegression(BaseEstimator):
    """
    Ridge Regression Estimator

    Solving Ridge Regression optimization problem
    """

    def __init__(self, lam: float,
                 include_intercept: bool = True) -> RidgeRegression:
        """
        Initialize a ridge regression model

        Parameters
        ----------
        lam: float
            Regularization parameter to be used when fitting a model

        include_intercept: bool, default=True
            Should fitted model include an intercept or not

        Attributes
        ----------
        include_intercept_: bool
            Should fitted model include an intercept or not

        coefs_: ndarray of shape (n_features,) or (n_features+1,)
            Coefficients vector fitted by linear regression. To be set in
            `LinearRegression.fit` function.
        """

        """
        Initialize a ridge regression model
        :param lam: scalar value of regularization parameter
        """
        super().__init__()
        self.coefs_ = None
        self.include_intercept_ = include_intercept
        self.lam_ = lam

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        Fit Ridge regression model to given samples

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to

        Notes
        -----
        Fits model with or without an intercept depending on value of `self.include_intercept_`
        """
        d = X.shape[1]
        lambda_mat = np.eye(d) * np.sqrt(self.lam_)

        if self.include_intercept_:
            #  make the first column of X be ones (include W_0)
            X = np.pad(X, [(0, 0), (1, 0)], mode="constant", constant_values=1)
            #  make the first column of lambda_mat be zeros
            #  (don't minimize the norm over w_0)
            lambda_mat = np.pad(lambda_mat, [(0, 0), (1, 0)], mode="constant",
                                constant_values=0)
        new_X = np.concatenate((X, lambda_mat), axis=0)
        new_y = np.concatenate((y, np.zeros(d)), axis=0)
        self.coefs_ = pinv(new_X) @ new_y
        self.fitted_ = True

    def _predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict responses for given samples using fitted estimator

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples
        """
        if self.coefs_ is None:
            raise Exception("not fitted model")

        if self.include_intercept_:
            #  make the first column of X be ones
            X = np.pad(X, [(0,0),(1,0)], mode="constant", constant_values=1)

        return X @ self.coefs_


    def _loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate performance under MSE loss function

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples

        y : ndarray of shape (n_samples, )
            True labels of test samples

        Returns
        -------
        loss : float
            Performance under MSE loss function
        """
        from IMLearn.metrics.loss_functions import mean_square_error
        if not self.fitted_:
            raise Exception("not fitted model")
        y_pred = self._predict(X)
        return mean_square_error(y, y_pred)
