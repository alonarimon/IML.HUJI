import numpy as np
import pandas as pd
from typing import Tuple, List, Callable, Type
from sklearn.metrics import roc_curve, auc

from IMLearn import BaseModule
from IMLearn.desent_methods import GradientDescent, FixedLR, ExponentialLR
from IMLearn.desent_methods.modules import L1, L2
from IMLearn.learners.classifiers.logistic_regression import LogisticRegression
from IMLearn.utils import split_train_test
from IMLearn.model_selection.cross_validate import cross_validate
from IMLearn.metrics.loss_functions import misclassification_error

import plotly.graph_objects as go


def plot_descent_path(module: Type[BaseModule],
                      descent_path: np.ndarray,
                      title: str = "",
                      xrange=(-1.5, 1.5),
                      yrange=(-1.5, 1.5)) -> go.Figure:
    """
    Plot the descent path of the gradient descent algorithm

    Parameters:
    -----------
    module: Type[BaseModule]
        Module type for which descent path is plotted

    descent_path: np.ndarray of shape (n_iterations, 2)
        Set of locations if 2D parameter space being the regularization path

    title: str, default=""
        Setting details to add to plot title

    xrange: Tuple[float, float], default=(-1.5, 1.5)
        Plot's x-axis range

    yrange: Tuple[float, float], default=(-1.5, 1.5)
        Plot's x-axis range

    Return:
    -------
    fig: go.Figure
        Plotly figure showing module's value in a grid of [xrange]x[yrange] over which regularization path is shown

    Example:
    --------
    fig = plot_descent_path(IMLearn.desent_methods.modules.L1, np.ndarray([[1,1],[0,0]]))
    fig.show()
    """

    def predict_(w):
        return np.array([module(weights=wi).compute_output() for wi in w])

    from utils import decision_surface
    return go.Figure([decision_surface(predict_, xrange=xrange, yrange=yrange,
                                       density=70, showscale=False),
                      go.Scatter(x=descent_path[:, 0], y=descent_path[:, 1],
                                 mode="markers+lines", marker_color="black")],
                     layout=go.Layout(xaxis=dict(range=xrange),
                                      yaxis=dict(range=yrange),
                                      title=f"GD Descent Path {title}"))


def get_gd_state_recorder_callback() -> Tuple[
    Callable[[], None], List[np.ndarray], List[np.ndarray]]:
    """
    Callback generator for the GradientDescent class, recording the objective's value and parameters at each iteration

    Return:
    -------
    callback: Callable[[], None]
        Callback function to be passed to the GradientDescent class, recoding the objective's value and parameters
        at each iteration of the algorithm

    values: List[np.ndarray]
        Recorded objective values

    weights: List[np.ndarray]
        Recorded parameters
    """
    vals = []
    ws = []

    def callback(solver=None, weights=None,
                 val=None, grad=None,
                 t=0, eta=0, delta=0):
        vals.append(val)
        ws.append(weights)

    return callback, vals, ws


def compare_fixed_learning_rates(
        init: np.ndarray = np.array([np.sqrt(2), np.e / 3]),
        etas: Tuple[float] = (1, .1, .01, .001)):
    #  Minimize the L1 and L2 modules for each of the following fixed lrs
    for eta in etas:
        lr = FixedLR(eta)
        for module in [(L1, "L1"), (L2, "L2")]:
            initial_weights = np.copy(init)
            f = module[0](initial_weights)
            callback, vals, ws = get_gd_state_recorder_callback()
            gd = GradientDescent(learning_rate=lr, callback=callback)
            f.weights = gd.fit(f, None, None)

            # Plot the descent trajectory for each of the settings
            fig = plot_descent_path(module[0], np.array(ws),
                                    title=f"base learning rate: {eta}, objective: {module[1]}")
            fig.show()

            # plot the convergence rate
            data = go.Scatter(x=np.arange(len(vals)), y=vals,
                              mode='markers', marker=dict(size=2))
            fig = go.Figure(data=data) \
                .update_layout(
                title_text=f"convergence rate with learning rate: {eta},"
                           f" objective: {module[1]}") \
                .update_xaxes(title_text="iteration number") \
                .update_yaxes(title_text="norm")
            fig.show()

            # the lowest loss achieved when minimizing each of the modules
            print(
                f"the loss achived for lr {eta}, objective {module[1]}: "
                f"{f.compute_output()}")
        print("-" * 10)


def compare_exponential_decay_rates(
        init: np.ndarray = np.array([np.sqrt(2), np.e / 3]),
        eta: float = .1,
        gammas: Tuple[float] = (.9, .95, .99, 1)):
    # Optimize the L1 objective using different decay-rate values of the exponentially decaying learning rate
    fig = go.Figure().update_layout(
        title_text=f"convergence rate with learning rate: {eta},"
                   f" objective: L1") \
        .update_xaxes(title_text="iteration number") \
        .update_yaxes(title_text="l1 norm")

    for gamma in gammas:
        lr = ExponentialLR(eta, gamma)
        initial_weights = np.copy(init)
        f = L1(initial_weights)
        callback, vals, ws = get_gd_state_recorder_callback()
        gd = GradientDescent(learning_rate=lr, callback=callback)
        _ = gd.fit(f, None, None)

        # add the convergence rate trace to the plot
        fig.add_trace(go.Scatter(x=np.arange(len(vals)), y=vals,
                                 mode="markers+lines",
                                 name=f"gamma: {gamma}",
                                 marker=dict(size=0.5),
                                 showlegend=True))

        # the lowest loss achieved when minimizing each of the modules
        print(f"the lowest loss achived for gamma {gamma}: "
              f"{np.min(np.array(vals))}")
        print("-" * 10)

    # Plot algorithm's convergence for the different values of gamma
    fig.show()

    # Plot descent path for gamma=0.95
    lr = ExponentialLR(eta, 0.95)
    initial_weights = np.copy(init)
    f = L1(initial_weights)
    callback, vals, ws = get_gd_state_recorder_callback()
    gd = GradientDescent(learning_rate=lr, callback=callback)
    _ = gd.fit(f, None, None)
    fig = plot_descent_path(L1, np.array(ws),
                            title=f"descent path for gamma = {0.95}")
    fig.show()


def load_data(path: str = "../datasets/SAheart.data",
              train_portion: float = .8) -> \
        Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Load South-Africa Heart Disease dataset and randomly split into a train- and test portion

    Parameters:
    -----------
    path: str, default= "../datasets/SAheart.data"
        Path to dataset

    train_portion: float, default=0.8
        Portion of dataset to use as a training set

    Return:
    -------
    train_X : DataFrame of shape (ceil(train_proportion * n_samples), n_features)
        Design matrix of train set

    train_y : Series of shape (ceil(train_proportion * n_samples), )
        Responses of training samples

    test_X : DataFrame of shape (floor((1-train_proportion) * n_samples), n_features)
        Design matrix of test set

    test_y : Series of shape (floor((1-train_proportion) * n_samples), )
        Responses of test samples
    """
    df = pd.read_csv(path)
    df.famhist = (df.famhist == 'Present').astype(int)
    return split_train_test(df.drop(['chd', 'row.names'], axis=1), df.chd,
                            train_portion)


def fit_logistic_regression():
    # Load and split SA Heard Disease dataset
    X_train, y_train, X_test, y_test = load_data()
    X_train, y_train, X_test, y_test = X_train.to_numpy(), y_train.to_numpy(), \
                                       X_test.to_numpy(), y_test.to_numpy()

    # Plotting convergence rate of logistic regression over SA heart disease data
    logistic_reg = LogisticRegression(
        solver=GradientDescent(learning_rate=FixedLR(1e-4), max_iter=20000))
    logistic_reg.fit(X_train, y_train)
    y_prob = logistic_reg.predict_proba(X_train)
    fpr, tpr, thresholds = roc_curve(y_train, y_prob)
    fig = go.Figure(
        data=[go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                         line=dict(color="black", dash='dash'),
                         name="Random Class Assignment"),
              go.Scatter(x=fpr, y=tpr, mode='markers+lines', text=thresholds,
                         name="", showlegend=False, marker_size=5,
                         marker_color="rgb(49,54,149)",
                         hovertemplate="<b>Threshold:</b>%{text:.3f}<br>FPR: %{x:.3f}<br>TPR: %{y:.3f}")],
        layout=go.Layout(
            title=rf"$\text{{ROC Curve Of Fitted Model - AUC}}={auc(fpr, tpr):.6f}$",
            xaxis=dict(title=r"$\text{False Positive Rate (FPR)}$"),
            yaxis=dict(title=r"$\text{True Positive Rate (TPR)}$")))
    fig.show()

    best_alpha = thresholds[np.argmax(tpr - fpr)]
    test_error = logistic_reg.loss(X_test, y_test)
    print(f"value of α that achieves the optimal ROC value : {best_alpha}\n"
          f"the model’s test error with that alpha : {test_error}")

    # Fitting l1- and l2-regularized logistic regression models, using cross-validation to specify values
    # of regularization parameter
    lambdas = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1]
    for penalty in ["l1", "l2"]:
        val_errors = np.zeros(len(lambdas))
        print("="*10, penalty, "="*10)
        for i in range(len(lambdas)):
            print("-"*10, lambdas[i], "-"*10)
            reg_logistic_regression = LogisticRegression(
                solver=GradientDescent(learning_rate=FixedLR(1e-4), max_iter=20000),
                penalty=penalty, alpha=0.5, lam=lambdas[i])
            _, val_errors[i] = cross_validate(reg_logistic_regression, X_train ,y_train,
                           misclassification_error)
        best_lam = lambdas[np.argmin(val_errors)]
        print(penalty + " best lambda : ", best_lam)
        reg_logistic_regression = LogisticRegression(
            solver=GradientDescent(learning_rate=FixedLR(1e-4), max_iter=20000),
            penalty=penalty, alpha=0.5, lam=best_lam)
        reg_logistic_regression.fit(X_train, y_train)
        print(f"the model’s test error with that lambda : "
              f"{reg_logistic_regression.loss(X_test, y_test)}")



if __name__ == '__main__':
    np.random.seed(0)
    compare_fixed_learning_rates()
    compare_exponential_decay_rates()
    fit_logistic_regression()
