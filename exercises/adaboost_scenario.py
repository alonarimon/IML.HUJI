import sys

import numpy as np
from typing import Tuple

import IMLearn.metrics
from IMLearn.metalearners.adaboost import AdaBoost
from IMLearn.learners.classifiers import DecisionStump
from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def decision_surface(predict, t, xrange, yrange, density=120, dotted=False,
                     colorscale=custom, showscale=True):
    xrange, yrange = np.linspace(*xrange, density), np.linspace(*yrange,
                                                                density)
    xx, yy = np.meshgrid(xrange, yrange)
    pred = predict(np.c_[xx.ravel(), yy.ravel()], t)

    if dotted:
        return go.Scatter(x=xx.ravel(), y=yy.ravel(), opacity=1,
                          mode="markers", marker=dict(color=pred, size=1,
                                                      colorscale=colorscale,
                                                      reversescale=False),
                          hoverinfo="skip", showlegend=False)
    return go.Contour(x=xrange, y=yrange, z=pred.reshape(xx.shape),
                      colorscale=colorscale, reversescale=False, opacity=.7,
                      connectgaps=True, hoverinfo="skip", showlegend=False,
                      showscale=showscale)


def generate_data(n: int, noise_ratio: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a dataset in R^2 of specified size

    Parameters
    ----------
    n: int
        Number of samples to generate

    noise_ratio: float
        Ratio of labels to invert

    Returns
    -------
    X: np.ndarray of shape (n_samples,2)
        Design matrix of samples

    y: np.ndarray of shape (n_samples,)
        Labels of samples
    """
    '''
    generate samples X with shape: (num_samples, 2) and labels y with shape (num_samples).
    num_samples: the number of samples to generate
    noise_ratio: invert the label for this ratio of the samples
    '''
    X, y = np.random.rand(n, 2) * 2 - 1, np.ones(n)
    y[np.sum(X ** 2, axis=1) < 0.5 ** 2] = -1
    y[np.random.choice(n, int(noise_ratio * n))] *= -1
    return X, y


def fit_and_evaluate_adaboost(noise, n_learners=250, train_size=5000,
                              test_size=500):
    (train_X, train_y), (test_X, test_y) = generate_data(train_size,
                                                         noise), generate_data(
        test_size, noise)

    # Question 1: Train- and test errors of AdaBoost in noiseless case
    ada = AdaBoost(DecisionStump, n_learners)
    ada._fit(train_X, train_y)

    train_error = np.zeros(n_learners)
    test_error = np.zeros(n_learners)
    for n in range(n_learners):
        train_error[n] = ada.partial_loss(train_X, train_y, n + 1)
        test_error[n] = ada.partial_loss(test_X, test_y, n + 1)

    x = np.arange(1, n_learners + 1)
    fig = go.Figure([
        go.Scatter(x=x, y=train_error, mode='markers + lines',
                   name=r'$Train$', line=dict(width=1)),
        go.Scatter(x=x, y=test_error, mode='markers + lines', name=r'$Test$',
                   line=dict(width=1))])
    fig.update_layout(title=rf"$\textbf{{Train and Test errors}}$",
                      xaxis=dict(title="number of fitted learners"),
                      yaxis=dict(title="Error"))
    fig.show()

    # Question 2: Plotting decision surfaces
    T = [(0, 5), (1, 50), (2, 100), (3, 250)]
    lims = np.array([np.r_[train_X, test_X].min(axis=0),
                     np.r_[train_X, test_X].max(axis=0)]).T + np.array(
        [-.1, .1])
    symbols = np.array(["circle", "x"])
    # translation of responses to 0/1
    train_y_zero_ones = np.zeros(train_y.shape[0], dtype=int)
    train_y_zero_ones[train_y == 1] = 1
    test_y_zero_ones = np.zeros(test_y.shape[0], dtype=int)
    test_y_zero_ones[test_y == 1] = 1

    fig = make_subplots(rows=2, cols=2,
                        subplot_titles=[f"{t[1]} iterations" for t in T],
                        horizontal_spacing=0.05, vertical_spacing=.2)

    for (i, t) in T:

        fig.add_traces([decision_surface(ada.partial_predict, t, lims[0],
                                         lims[1], showscale=False),
                        go.Scatter(x=test_X[:, 0], y=test_X[:, 1],
                                   mode="markers", showlegend=False,
                                   marker=dict(color=test_y, symbol=symbols[
                                       test_y_zero_ones],
                                               colorscale=[custom[0],
                                                           custom[-1]],
                                               line=dict(color="black",
                                                         width=1)))],

                       rows=(i // 2) + 1, cols=(i % 2) + 1)

    fig.update_layout(title=rf"$\textbf{{Decision Boundaries Of Ensembles}}$",
                      margin=dict(t=100))
    fig.show()

    # Question 3: Decision surface of best performing ensemble
    best_ensemble_size = np.argmin(test_error) + 1
    best_accuracy = IMLearn.metrics.accuracy(
        ada.partial_predict(test_X, best_ensemble_size), test_y)
    decision_boundries = decision_surface(ada.partial_predict,
                                          best_ensemble_size, lims[0],
                                          lims[1], showscale=False)
    fig = go.Figure([decision_boundries,
                     go.Scatter(x=test_X[:, 0], y=test_X[:, 1],
                                mode="markers", showlegend=False,
                                marker=dict(color=test_y, symbol=symbols[
                                    test_y_zero_ones],
                                            colorscale=[custom[0],
                                                        custom[-1]],
                                            line=dict(color="black",
                                                      width=1)))])

    fig.update_layout(
        title=rf"$\textbf{{{best_ensemble_size} learners Ensemble. Accuracy: {best_accuracy}}}$",
        margin=dict(t=100))
    fig.show()
    # Question 4: Decision surface with weighted samples
    decision_boundries = decision_surface(ada.partial_predict, n_learners,
                                          lims[0],
                                          lims[1], showscale=False)
    D = (ada.D_ / np.max(ada.D_)) * 5


    fig = go.Figure([decision_boundries,
                     go.Scatter(x=train_X[:, 0], y=train_X[:, 1],
                                mode="markers", showlegend=False,
                                marker=dict(color=train_y, symbol=symbols[
                                    train_y_zero_ones], size=D,
                                            colorscale=[custom[0],
                                                        custom[-1]],
                                            line=dict(color="black",
                                                      width=1)))])

    fig.update_layout(title=rf"$\textbf{{Weighted Training Samples}}$",
                      margin=dict(t=100))
    fig.show()


if __name__ == '__main__':
    np.random.seed(0)
    fit_and_evaluate_adaboost(noise=0)
    fit_and_evaluate_adaboost(noise=0.4)
