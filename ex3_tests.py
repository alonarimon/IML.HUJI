import numpy as np
from IMLearn.metrics import loss_functions
from IMLearn.learners.classifiers import linear_discriminant_analysis as LDA
import os.path

from IMLearn.learners.classifiers import Perceptron, LDA, GaussianNaiveBayes
from typing import Tuple
from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from math import atan2, pi

datasets_path = r".\datasets"  # todo: working on linux?


def load_dataset(filename: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load dataset for comparing the Gaussian Naive Bayes and LDA classifiers. File is assumed to be an
    ndarray of shape (n_samples, 3) where the first 2 columns represent features and the third column the class

    Parameters
    ----------
    filename: str
        Path to .npy data file

    Returns
    -------
    X: ndarray of shape (n_samples, 2)
        Design matrix to be used

    y: ndarray of shape (n_samples,)
        Class vector specifying for each sample its class

    """
    data = np.load(filename)
    return data[:, :2], data[:, 2].astype(int)


def test_miss():
    a = np.arange(10)
    # b = np.array([1,3,0,4,6,4,1,0,0,1])
    b = np.array([0, 1, 2, 0, 4, 5, 0, 7, 0, 9])
    print(a)
    print(b)
    print(a != b)
    print(type(np.sum(a != b) / a.shape[0]))
    print(loss_functions.misclassification_error(a, b))
    print(loss_functions.misclassification_error(a, b, True))
    print(loss_functions.misclassification_error(a, b, False))
    print(type(loss_functions.misclassification_error(a, b, True)))
    print(type(loss_functions.misclassification_error(a, b, False)))


def test_accuracy():
    a = np.arange(10)
    # b = np.array([1,3,0,4,6,4,1,0,0,1])
    b = np.array([0, 1, 2, 0, 4, 5, 0, 7, 9])

    print(loss_functions.accuracy(a, b))
    print(type(loss_functions.accuracy(a, b)))


def test_fit(X, y):
    lda = LDA.LDA()
    lda.fit(X, y)
    print(lda.classes_, "\n")
    print(lda.mu_, "\n")
    print(lda.cov_, "\n")
    print(lda._cov_inv, "\n")
    print(lda.pi_)


def test_predict(X, y, X_to_pred):
    lda = LDA.LDA()
    lda.fit(X, y)
    print(lda.predict(X_to_pred))


def compare_gaussian_classifiers():
    """
    Fit both Gaussian Naive Bayes and LDA classifiers on both gaussians1 and gaussians2 datasets
    """
    for f in ["gaussian1.npy"]:
        # Load dataset
        X, y = load_dataset(os.path.join(datasets_path, f))

        # Fit models and predict over training set
        lda = LDA()
        gaussian_naive = GaussianNaiveBayes()
        lda._fit(X, y)
        gaussian_naive._fit(X, y)

        lda_mu = lda.mu_
        lda_cov = lda.cov_
        g_mu = gaussian_naive.mu_
        g_vars = gaussian_naive.vars_
        print("-" * 10 + " lda mu " + "-" * 10)
        print(lda_mu)
        print("-" * 10 + " lda cov " + "-" * 10)
        print(lda_cov)
        print("-" * 10 + " gaussian naive mu " + "-" * 10)
        print(g_mu)
        print("-" * 10 + " gaussian naive vars " + "-" * 10)
        print(g_vars)


        lda_likelihood = lda.likelihood(X)
        lda_predict = lda._predict(X)
        lda_argmax_likelihood = np.argmax(lda_likelihood, axis=1)

        print("-" * 10 + "lda_likelihood" + "-" * 10)
        # print(lda_likelihood)
        print(lda_likelihood.shape)
        print("-" * 10 + "lda_predict" + "-" * 10)
        print(lda_predict.shape)
        print("-" * 10 + "lda_argmax_likelihood" + "-" * 10)
        print(lda_argmax_likelihood.shape)
        print("-" * 10, "lda_predict == lda_argmax", "-" * 10)
        print((lda_predict == lda_argmax_likelihood).all())
        print("-" * 10, "lda_loss(X, y)", "-" * 10)
        print(lda._loss(X, y))

        g_likelihood = gaussian_naive.likelihood(X)
        g_predict = gaussian_naive._predict(X)
        g_argmax_likelihood = np.argmax(g_likelihood, axis=1)

        print("-" * 10 + "g_likelihood" + "-" * 10)
        # print(g_likelihood)
        print(g_likelihood.shape)
        print("-" * 10 + "g_predict" + "-" * 10)
        print(g_predict.shape)
        print("-" * 10 + "g_argmax_likelihood" + "-" * 10)
        print(g_argmax_likelihood.shape)
        print("-" * 10, "g_predict == g_argmax", "-" * 10)
        print((g_predict == g_argmax_likelihood).all())
        print("-" * 10, "g_loss(X, y)", "-" * 10)
        print(gaussian_naive._loss(X, y))

        # todo: test with string classes (if needed at all)
        # Plot a figure with two suplots, showing the Gaussian Naive Bayes predictions on the left and LDA predictions
        # on the right. Plot title should specify dataset used and subplot titles should specify algorithm and accuracy
        # Create subplots
        from IMLearn.metrics import accuracy
        # raise NotImplementedError()
        #
        # # Add traces for data-points setting symbols and colors
        # raise NotImplementedError()
        #
        # # Add `X` dots specifying fitted Gaussians' means
        # raise NotImplementedError()
        #
        # # Add ellipses depicting the covariances of the fitted Gaussians
        # raise NotImplementedError()


if __name__ == '__main__':
    S=[([1,1],0),([1,2],0),([2,3],1),([2,4],1),([3,3],1),([3,4],1)]
    X = np.array([tup[0] for tup in S])
    y = np.array([tup[1] for tup in S])

    g = GaussianNaiveBayes()

    g._fit(X, y)
    print(X)
    print(y)
    print("-"*30)
    # print(g.pi_)
    # print(g.mu_)
    print(g.vars_)

