from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn import datasets
from IMLearn.metrics import mean_square_error
from IMLearn.utils import split_train_test
from IMLearn.model_selection import cross_validate
from IMLearn.learners.regressors import PolynomialFitting, LinearRegression, \
    RidgeRegression
from sklearn.linear_model import Lasso

from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def target_func(X):
    return (X + 3) * (X + 2) * (X + 1) * (X - 1) * (X - 2)


def load_poly_dataset(n_samples: int = 100, noise: float = 5):
    """
    Generate dataset for model f(x)=(x+3)(x+2)(x+1)(x-1)(x-2) + eps for eps Gaussian noise
    :param n_samples:
    :param noise:
    :return: X, y, y_noiseless:
      X : DataFrame of shape (n_samples, n_features)
        Data frame of samples and feature values.

      y : Series of shape (n_samples, )
        Responses corresponding samples in data frame.

    """
    X = np.linspace(-1.2, 2.0, num=n_samples)
    eps = np.random.normal(0, noise, n_samples)
    y = target_func(X) + eps

    return X, y


def select_polynomial_degree(n_samples: int = 100, noise: float = 5):
    """
    Simulate data from a polynomial model and use cross-validation to select the best fitting degree

    Parameters
    ----------
    n_samples: int, default=100
        Number of samples to generate

    noise: float, default = 5
        Noise level to simulate in responses
    """
    # Question 1 - Generate dataset for model f(x)=(x+3)(x+2)(x+1)(x-1)(x-2) + eps for eps Gaussian noise
    # and split into training- and testing portions
    X, y = load_poly_dataset(n_samples, noise)
    df_train_X, df_train_y, df_test_X, df_test_y = split_train_test(
        pd.DataFrame(X.reshape(n_samples, 1), columns=['x value']),
        pd.Series(y),
        train_proportion=(
                2 / 3))
    # convert to numpy
    train_X = df_train_X.to_numpy().flatten()
    train_y = df_train_y.to_numpy()
    test_X = df_test_X.to_numpy().flatten()
    test_y = df_test_y.to_numpy()

    y_noiseless = target_func(X)

    # plot
    fig = go.Figure([
        go.Scatter(x=X, y=y_noiseless, mode="markers",
                   name="True model (noiseless)",
                   marker=dict(color="black", opacity=.7), showlegend=True),

        go.Scatter(x=train_X, y=train_y, mode="markers", name="train set",
                   marker=dict(color="blue", opacity=.7), showlegend=True),
        go.Scatter(x=test_X, y=test_y, mode="markers", name="test set",
                   marker=dict(color="red", opacity=.7), showlegend=True)])

    fig.update_layout(title=rf"$\textbf{{test and train sets}}$",
                      xaxis=dict(title="x"),
                      yaxis=dict(title="f(x)"))

    fig.show()

    # Question 2 - Perform CV for polynomial fitting with degrees 0,1,...,10
    training_errors = np.zeros(11)
    validation_errors = np.zeros(11)
    for k in range(11):
        poly_k = PolynomialFitting(k)
        training_errors[k], validation_errors[k] = cross_validate(poly_k,
                                                                  train_X,
                                                                  train_y,
                                                                  mean_square_error)
    fig = go.Figure([
        go.Scatter(x=np.arange(11), y=training_errors, mode="markers+lines",
                   name="average training errors",
                   marker=dict(color="blue", opacity=.7), showlegend=True),
        go.Scatter(x=np.arange(11), y=validation_errors, mode="markers+lines",
                   name="average validation errors",
                   marker=dict(color="red", opacity=.7), showlegend=True)
    ])

    fig.update_layout(title=rf"$\textbf{{average errors per degree}}$",
                      xaxis=dict(title="the polynom degree (k)"),
                      yaxis=dict(title="errors"))

    fig.show()

    # Question 3 - Using best value of k, fit a k-degree polynomial model and report test error
    best_k = np.argmin(validation_errors)
    best_validation_error = validation_errors[best_k]
    poly = PolynomialFitting(best_k)
    poly.fit(train_X, train_y)
    test_error = poly.loss(test_X, test_y)
    print(f"noise: {noise}\n"
          f"best k: {best_k} \n"
          f"best val error: {best_validation_error}\n"
          f"test error: {np.round(test_error, 2)}\n")


# ======================= regularization selection ============================

def load_diabetes_dataset(n_samples):
    X, y = datasets.load_diabetes(return_X_y=True, as_frame=True)
    X = X.to_numpy()
    y = y.to_numpy()
    train_X = X[:n_samples+1]
    train_y = y[:n_samples+1]
    test_X = X[n_samples+1:]
    test_y = y[n_samples+1:]


    return train_X, train_y, test_X, test_y


def select_regularization_parameter(n_samples: int = 50,
                                    n_evaluations: int = 500,
                                    lambda_range_start: float = 0,
                                    lambda_range_end: float = 1):
    """
    Using sklearn's diabetes dataset use cross-validation to select the best fitting regularization parameter
    values for Ridge and Lasso regressions

    Parameters
    ----------
    n_samples: int, default=50
        Number of samples to generate

    n_evaluations: int, default = 500
        Number of regularization parameter values to evaluate for each of the algorithms
    """
    # Question 6 - Load diabetes dataset and split into training and testing portions
    train_X, train_y, test_X, test_y = load_diabetes_dataset(n_samples)

    # Question 7 - Perform CV for different values of the regularization parameter for Ridge and Lasso regressions
    lambda_values = np.linspace(lambda_range_start, lambda_range_end,
                                n_evaluations)

    ridge_training_errors = np.zeros(n_evaluations)
    ridge_validation_errors = np.zeros(n_evaluations)
    lasso_training_errors = np.zeros(n_evaluations)
    lasso_validation_errors = np.zeros(n_evaluations)

    for i in range(n_evaluations):
        ridge = RidgeRegression(lambda_values[i])
        lasso = Lasso(lambda_values[i])

        ridge_training_errors[i], ridge_validation_errors[i] = cross_validate(
            ridge, train_X, train_y,
            mean_square_error)
        lasso_training_errors[i], lasso_validation_errors[i] = cross_validate(
            lasso, train_X, train_y,
            mean_square_error)

    fig = make_subplots(1, 2, subplot_titles=["ridge", "lasso"],
                        horizontal_spacing=0.2, x_title="lambda value",
                        y_title="errors")
    fig.add_traces([go.Scatter(x=lambda_values, y=ridge_training_errors,
                               mode="markers+lines",
                               name="average training errors",
                               marker=dict(color="blue", opacity=.7, size=1),
                               showlegend=True),
                    go.Scatter(x=lambda_values, y=ridge_validation_errors,
                               mode="markers+lines",
                               name="average validation errors",
                               marker=dict(color="red", opacity=.7, size=1),
                               showlegend=True)
                    ], rows=1, cols=1)
    fig.add_traces([go.Scatter(x=lambda_values, y=lasso_training_errors,
                               mode="markers+lines",
                               name="average training errors",
                               marker=dict(color="blue", opacity=.7, size=1),
                               showlegend=False),
                    go.Scatter(x=lambda_values, y=lasso_validation_errors,
                               mode="markers+lines",
                               name="average validation errors",
                               marker=dict(color="red", opacity=.7, size=1),
                               showlegend=False)
                    ], rows=1, cols=2)

    fig.update_layout(
        title=rf"$\textbf{{errors as a function of the tested regularization parameter value}}$")

    fig.show()

    # Question 8 - Compare best Ridge model, best Lasso model and Least Squares model
    best_ridge_lambda = lambda_values[np.argmin(ridge_validation_errors)]
    best_lasso_lambda = lambda_values[np.argmin(lasso_validation_errors)]
    print("-"*20)
    print(f"best ridge lambda: {best_ridge_lambda}\n"
          f"best lasso lambda: {best_lasso_lambda}")
    print("-"*20)

    best_ridge = RidgeRegression(best_ridge_lambda)
    best_ridge.fit(train_X, train_y)
    ridge_test_error = best_ridge.loss(test_X, test_y)

    best_lasso = Lasso(best_lasso_lambda)
    best_lasso.fit(train_X, train_y)
    lasso_pred = best_lasso.predict(test_X)
    lasso_test_error = mean_square_error(test_y, lasso_pred)

    least_squares = LinearRegression()
    least_squares.fit(train_X, train_y)
    ls_test_error = least_squares.loss(test_X, test_y)

    print(f"ridge test error: {ridge_test_error}\n"
          f"lasso test error: {lasso_test_error}\n"
          f"LS test error: {ls_test_error}")




if __name__ == '__main__':
    np.random.seed(0)
    # select_polynomial_degree(n_samples=100, noise=5)
    # select_polynomial_degree(n_samples=100, noise=0)
    # select_polynomial_degree(n_samples=1500, noise=10)

    # select_regularization_parameter(lambda_range_start=0, lambda_range_end=1)
    # select_regularization_parameter(lambda_range_start=0, lambda_range_end=0.5)
    # select_regularization_parameter(lambda_range_start=0, lambda_range_end=20)
    select_regularization_parameter(lambda_range_start=0, lambda_range_end=5)

