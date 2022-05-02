from typing import NoReturn
from ...base import BaseEstimator
import numpy as np
from numpy.linalg import det, inv


class LDA(BaseEstimator):
    """
    Linear Discriminant Analysis (LDA) classifier

    Attributes
    ----------
    self.classes_ : np.ndarray of shape (n_classes,)
        The different labels classes. To be set in `LDA.fit`

    self.mu_ : np.ndarray of shape (n_classes,n_features)
        The estimated features means for each class. To be set in `LDA.fit`

    self.cov_ : np.ndarray of shape (n_features,n_features)
        The estimated features covariance. To be set in `LDA.fit`

    self._cov_inv : np.ndarray of shape (n_features,n_features)
        The inverse of the estimated features covariance. To be set in `LDA.fit`

    self.pi_: np.ndarray of shape (n_classes)
        The estimated class probabilities. To be set in `GaussianNaiveBayes.fit`
    """

    def __init__(self):
        """
        Instantiate an LDA classifier
        """
        super().__init__()
        self.classes_, self.mu_, self.cov_, self._cov_inv, self.pi_ = None, None, None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits an LDA model.
        Estimates gaussian for each label class - Different mean vector, same covariance
        matrix with dependent features.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        # classes_ = The different labels classes.
        # n[k] = the number of samples holding the k'th label
        self.classes_, inverse, n = np.unique(y, return_inverse=True,
                                              return_counts=True)

        self.mu_ = self.MLE_mu(X, y, self.classes_)
        self.cov_ = self.unbiased_cov(X, self.mu_, inverse)
        self._cov_inv = inv(self.cov_)
        self.pi_ = n / len(X)
        self.fitted_ = True

    def MLE_mu(self, X, y, classes) -> np.ndarray:
        """
        calculates the estimated features means for each class.
        according to the MLE principle as learned in class.
        :param classes: The different labels classes.
        :param X:
        :param y:
        :return: ndarray with shape (n_classes,n_features)
        """  # todo: without for-loop?
        return np.array(
            [X[y == classes[k]].mean(axis=0) for k in range(classes.shape[0])])

    def unbiased_cov(self, X, mu, inverse) -> np.ndarray:
        """
        calculates the estimated features covariance.
        according to the MLE principle as learned in class.
        observe that here we take the unbiased form of the cov.
        :param inverse: ndarray with shape (n_samples)- for each y, 
                        the index of its class.
        :param mu:
        :param X:
        :return: np.ndarray of shape (n_features,n_features)
        """
        m = len(X)
        K = self.classes_.shape[0]
        mu_per_sample = np.take(mu, indices=inverse, axis=0)
        return np.dot((X - mu_per_sample).T, (X - mu_per_sample)) / (m - K)

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
        if not self.fitted_:
            raise ValueError(
                "Estimator must first be fitted before calling `predict` function")

        num_classes = self.pi_.shape[0]
        # output is of shape (num_classes, num_samples)
        output = (self.mu_ @ self._cov_inv @ X.T) + \
                 (np.log(self.pi_).reshape((num_classes, 1))) - \
                 ((0.5 * (np.diagonal(
                     self.mu_ @ self._cov_inv @ self.mu_.T)).reshape(
                     (num_classes, 1))))
        classes_index_predicted = np.argmax(output, axis=0)  # (n_samples,)
        pred = np.take(self.classes_, indices=classes_index_predicted)
        return pred

    def likelihood(self, X: np.ndarray) -> np.ndarray:
        """ #todo: log likelihood? and then like predict? make sure dont have computational issues
        Calculate the likelihood of a given data over the estimated model

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input data to calculate its likelihood over the different classes.

        Returns
        -------
        likelihoods : np.ndarray of shape (n_samples, n_classes)
            The likelihood for each sample under each of the classes

        """
        if not self.fitted_:
            raise ValueError(
                "Estimator must first be fitted before calling `likelihood` function")
        m = X.shape[0]
        d = X.shape[1]
        K = self.classes_.shape[0]
        likelihoods = np.zeros((m, K))
        for i in range(m):
            likelihoods[i] = self.sample_likelihood(X[i], d)
        return likelihoods

    def sample_likelihood(self, sample, d):
        normalized_x = sample - self.mu_  # shape = (num_classes, num_features)
        constant = 1/(np.sqrt(((2 * np.pi)**d) * det(self.cov_)))
        return constant * self.pi_ * np.exp(
            (-0.5)* np.diagonal(normalized_x @ self._cov_inv @ normalized_x.T))


    def _loss(self, X: np.ndarray, y: np.ndarray) -> float:
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
        from ...metrics import misclassification_error
        y_pred = self._predict(X)
        return misclassification_error(y_true=y, y_pred=y_pred) #todo: normalize?
