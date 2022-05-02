from typing import NoReturn
from ...base import BaseEstimator
import numpy as np
from numpy.linalg import det, inv


class GaussianNaiveBayes(BaseEstimator):
    """
    Gaussian Naive-Bayes classifier
    """
    def __init__(self):
        """
        Instantiate a Gaussian Naive Bayes classifier

        Attributes
        ----------
        self.classes_ : np.ndarray of shape (n_classes,)
            The different labels classes. To be set in `GaussianNaiveBayes.fit`

        self.mu_ : np.ndarray of shape (n_classes,n_features)
            The estimated features means for each class. To be set in `GaussianNaiveBayes.fit`

        self.vars_ : np.ndarray of shape (n_classes, n_features)
            The estimated features variances for each class. To be set in `GaussianNaiveBayes.fit`

        self.pi_: np.ndarray of shape (n_classes)
            The estimated class probabilities. To be set in `GaussianNaiveBayes.fit`
        """
        super().__init__()
        self.classes_, self.mu_, self.vars_, self.pi_ = None, None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits a gaussian naive bayes model

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
        self.vars_ = self.unbiased_vars(X, y, self.mu_, inverse, n)
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

    def unbiased_vars(self, X, y, mu, inverse, n) -> np.ndarray:
        """
        calculate the estimated variances of each feature and each class,
        and returns the unbiased estimator.
        :param inverse: ndarray with shape (n_samples)- for each y,
                        the index of its class.
        :param mu:
        :param X:
        :return: np.ndarray of shape (n_classes, n_features)
        """
        K = self.classes_.shape[0]
        mu_per_sample = np.take(mu, indices=inverse, axis=0)
        # inverse == k <-> y == classes[k]
        vars = np.array([
            np.sum((X[inverse == k]-mu_per_sample[inverse == k])**2,
                         axis=0)/(n[k])
            for k in range(K)])

        return vars


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
            raise ValueError("Estimator must first be fitted before calling `predict` function")

        classes_index_predicted = np.argmax(self.likelihood(X), axis=1)  # (n_samples,)
        # get the actual classes values
        pred = np.take(self.classes_, indices=classes_index_predicted)
        return pred

    def likelihood(self, X: np.ndarray) -> np.ndarray:
        """
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
            raise ValueError("Estimator must first be fitted before calling `likelihood` function")

        m = X.shape[0]
        d = X.shape[1]
        K = self.classes_.shape[0]
        likelihoods = np.zeros((m, K))
        for k in range(K):
            likelihoods[:, k] = self.k_likelihood(X, k, d)
        return likelihoods

    def k_likelihood(self, X, k, d):
        """
        calculate and return the likelihood for getting the k'th response
         for each sample (actually the posterior)
        :param X: shape (m, d)
        :param k: the class index
        :return: ndarray with shape (num_samples,)
        with k- likelihood for each one
        """
        normalized_X = X - self.mu_[k]  # shape = (num_samples, num_features)
        cov_k = np.diag(self.vars_[k])
        constant = 1/(np.sqrt(((2 * np.pi)**d) * det(cov_k)))
        likelihood = constant * self.pi_[k] * np.exp(
            (-0.5) * np.diagonal(normalized_X @ inv(cov_k) @ normalized_X.T))
        return likelihood


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
        return misclassification_error(y_true=y, y_pred=y_pred)

