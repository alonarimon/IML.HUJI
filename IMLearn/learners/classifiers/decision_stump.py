from __future__ import annotations
from typing import Tuple, NoReturn
from ...base import BaseEstimator
import numpy as np
from itertools import product
from IMLearn.metrics.loss_functions import misclassification_error


class DecisionStump(BaseEstimator):
    """
    A decision stump classifier for {-1,1} labels according to the CART algorithm

    Attributes
    ----------
    self.threshold_ : float
        The threshold by which the data is split

    self.j_ : int
        The index of the feature by which to split the data

    self.sign_: int
        The label to predict for samples where the value of the j'th feature is about the threshold
    """

    def __init__(self) -> DecisionStump:
        """
        Instantiate a Decision stump classifier
        """
        super().__init__()
        self.threshold_, self.j_, self.sign_ = None, None, None

    def weighted_loss(self, weighted_true_y, predicted_y) -> float:
        """
        weighted loss function,
        which takes into account the weight of the misclassified sample.
        So according to np.sign(y) and the predicted y you will know if you
        correctly classified or misclassified,
        and then use the magnitude of y (which is actually the value in D)
        to weight this mis-classification.
        :param weighted_true_y: ndarray of shape (n_samples, )
        :param predicted_y: ndarray of shape (n_samples, )
        :return: weighted loss type float
        """
        m = predicted_y.shape[0]
        mistakes_indexes = (weighted_true_y * predicted_y) < 0
        mistakes = np.abs(weighted_true_y[mistakes_indexes])

        return float(np.sum(mistakes) / m) #todo: with the normalization? (/m?)


    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits a decision stump to the given data

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """

        sign = -1
        j = 0
        threshold, threshold_err = self._find_threshold(X[:, j], y, sign)

        for cur_j in range(X.shape[1]):
            for cur_sign in [-1, 1]:
                cur_threshold, cur_threshold_err = self._find_threshold(X[:, cur_j], y,
                                                                        cur_sign)
                if cur_threshold_err < threshold_err:
                    threshold = cur_threshold
                    threshold_err = cur_threshold_err
                    sign = cur_sign
                    j = cur_j

        self.threshold_ = threshold
        self.j_ = j
        self.sign_ = sign

        self.fitted_ = True


    def _predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict responses for given samples using fitted estimator

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples

        Notes
        -----
        Feature values strictly below threshold are predicted as `-sign` whereas values which equal
        to or above the threshold are predicted as `sign`
        """
        if not self.fitted_:
            raise ValueError("Estimator must first be fitted before calling `predict` function")

        values = X[:, self.j_]  # the selected feature values
        predicted = np.zeros(X.shape[0])
        predicted[values >= self.threshold_] = self.sign_
        predicted[values < self.threshold_] = -self.sign_
        return predicted

    def _find_threshold(self, values: np.ndarray, labels: np.ndarray,
                        sign: int) -> Tuple[float, float]:
        """
        Given a feature vector and labels, find a threshold by which to perform a split
        The threshold is found according to the value minimizing the misclassification
        error along this feature

        Parameters
        ----------
        values: ndarray of shape (n_samples,)
            A feature vector to find a splitting threshold for

        labels: ndarray of shape (n_samples,)
            The labels to compare against

        sign: int
            Predicted label assigned to values equal to or above threshold

        Returns
        -------
        thr: float
            Threshold by which to perform split

        thr_err: float between 0 and 1
            Misclassificaiton error of returned threshold

        Notes
        -----
        For every tested threshold, values strictly below threshold are predicted
        as `-sign` whereas values
        which equal to or above the threshold are predicted as `sign`
        """
        m = values.shape[0]

        unique_values = np.unique(values)
        thresholds_errors = np.zeros(unique_values.shape[0])

        for i, v in np.ndenumerate(unique_values):
            # get the loss for each threshold value
            predicted = np.zeros(m)
            predicted[values >= v] = sign
            predicted[values < v] = -sign
            thresholds_errors[i[0]] = self.weighted_loss(labels, predicted)

        # get the index of the minimal value loss
        thr_index = np.argmin(thresholds_errors)
        return float(unique_values[thr_index]), thresholds_errors[thr_index]

    def _loss(self, X: np.ndarray, y: np.ndarray) -> float: #todo: change becaouse of weights?
        """
        Evaluate performance under misclassification loss function

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples

        y : ndarray of shape (n_samples, )
            True labels of test samples

        Returns
        -------
        loss : float
            Performance under missclassification loss function
        """
        return misclassification_error(y, self._predict(X))
