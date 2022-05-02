import os.path

from IMLearn.learners.classifiers import Perceptron, LDA, GaussianNaiveBayes
from typing import Tuple
from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from math import atan2, pi

datasets_path = r"..\datasets"  # todo: working on linux?


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


def run_perceptron():
    """
    Fit and plot fit progression of the Perceptron algorithm over both the linearly separable and inseparable datasets

    Create a line plot that shows the perceptron algorithm's training loss values (y-axis)
    as a function of the training iterations (x-axis).
    """
    for n, f in [("Linearly Separable", "linearly_separable.npy"),
                 ("Linearly Inseparable", "linearly_inseparable.npy")]:
        # Load dataset
        X, y = load_dataset(os.path.join(datasets_path, f))

        # Fit Perceptron and record loss in each fit iteration
        losses = []

        def call(perceptron: Perceptron, x_, y_):
            losses.append(perceptron._loss(X, y))

        p = Perceptron(callback=call)  # todo: intersept? max_iter?
        p._fit(X, y)

        # Plot figure of loss as function of fitting iteration
        data = go.Scatter(x=np.arange(len(losses)), y=losses,
                          mode='markers+lines', marker=dict(size=2))
        fig = go.Figure(data=data) \
            .update_layout(
            title_text="loss as function of fitting iteration: " + n) \
            .update_xaxes(title_text="fitting iteration") \
            .update_yaxes(title_text="loss")
        fig.show()


def get_ellipse(mu: np.ndarray, cov: np.ndarray):
    """
    Draw an ellipse centered at given location and according to specified covariance matrix

    Parameters
    ----------
    mu : ndarray of shape (2,)
        Center of ellipse

    cov: ndarray of shape (2,2)
        Covariance of Gaussian

    Returns
    -------
        scatter: A plotly trace object of the ellipse
    """
    l1, l2 = tuple(np.linalg.eigvalsh(cov)[::-1])
    theta = atan2(l1 - cov[0, 0], cov[0, 1]) if cov[0, 1] != 0 else (
        np.pi / 2 if cov[0, 0] < cov[1, 1] else 0)
    t = np.linspace(0, 2 * pi, 100)
    xs = (l1 * np.cos(theta) * np.cos(t)) - (l2 * np.sin(theta) * np.sin(t))
    ys = (l1 * np.sin(theta) * np.cos(t)) + (l2 * np.cos(theta) * np.sin(t))

    return go.Scatter(x=mu[0] + xs, y=mu[1] + ys, mode="lines",
                      showlegend=False,
                      marker_color="black")


def compare_gaussian_classifiers():
    """
    Fit both Gaussian Naive Bayes and LDA classifiers on both gaussians1 and gaussians2 datasets
    """
    for f in ["gaussian1.npy", "gaussian2.npy"]:
        # Load dataset
        X, y = load_dataset(os.path.join(datasets_path, f))

        # Fit models and predict over training set
        lda = LDA()
        gaussian_naive = GaussianNaiveBayes()
        lda._fit(X, y)
        gaussian_naive._fit(X, y)

        # Plot a figure with two suplots, showing the Gaussian Naive Bayes predictions on the left and LDA predictions
        # on the right. Plot title should specify dataset used and subplot titles should specify algorithm and accuracy

        # calculate accuracy
        from IMLearn.metrics import accuracy
        lda_pred = lda._predict(X)
        gaussian_naive_pred = gaussian_naive._predict(X)
        lda_accuracy = accuracy(y, lda_pred)
        g_naive_accuracy = accuracy(y, gaussian_naive_pred)

        # Create subplots
        fig = make_subplots(rows=1, cols=2,
                            subplot_titles=(
                                f"Gaussian Naive Bayes predictions.<br>"
                                f" accuracy : {g_naive_accuracy}"
                                , f"LDA predictions.<br>"
                                  f" accuracy : {lda_accuracy}"),
                            horizontal_spacing=0.2) \
            .update_layout(title=f + " dataset",
                           width=600, height=550, margin=dict(t=200))

        # Add traces for data-points setting symbols and colors
        gaussian_naive_data = go.Scatter(x=X[:, 0], y=X[:, 1],
                                         mode="markers",
                                         showlegend=False,
                                         marker=dict(color=gaussian_naive_pred,
                                                     symbol=y,
                                                     size=4))
        lda_data = go.Scatter(x=X[:, 0], y=X[:, 1],
                              mode="markers",
                              showlegend=False,
                              marker=dict(color=lda_pred,
                                          symbol=y,
                                          size=4))
        fig.add_trace(gaussian_naive_data, row=1, col=1) \
            .add_trace(lda_data, row=1, col=2)
        fig.update_xaxes(title_text="feature 0"). \
            update_yaxes(title_text="feature 1")

        # Add `X` dots specifying fitted Gaussians' means
        gaussian_naive_centers = go.Scatter(x=gaussian_naive.mu_[:, 0],
                                            y=gaussian_naive.mu_[:, 1],
                                            mode="markers",
                                            showlegend=False,
                                            marker=dict(color='black',
                                                        symbol='x',
                                                        size=8))
        lda_centers = go.Scatter(x=lda.mu_[:, 0], y=lda.mu_[:, 1],
                                 mode="markers",
                                 showlegend=False,
                                 marker=dict(color='black',
                                             symbol='x',
                                             size=8))
        fig.add_trace(gaussian_naive_centers, row=1, col=1). \
            add_trace(lda_centers, row=1, col=2)

        # # Add ellipses depicting the covariances of the fitted Gaussians
        for k in range(lda.classes_.shape[0]):
            fig.add_trace(get_ellipse(gaussian_naive.mu_[k],
                                      np.diag(gaussian_naive.vars_[k])), row=1,
                          col=1). \
                add_trace(get_ellipse(lda.mu_[k], lda.cov_), row=1, col=2)
        fig.show()


if __name__ == '__main__':
    np.random.seed(0)
    run_perceptron()
    compare_gaussian_classifiers()
