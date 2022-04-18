import math

import numpy as np
import pandas as pd


from IMLearn.metrics.loss_functions import mean_square_error
from sklearn.metrics import mean_squared_error as sklearn_mse
from IMLearn.learners.regressors.linear_regression import LinearRegression
from sklearn.linear_model import LinearRegression as LR

from IMLearn.utils import utils
from exercises import house_price_prediction

y_true = np.arange(10)
y1_pred = np.arange(10)+2
y2_pred = np.array([3,5,1,22,90,88,-15,-2, 0,444])

X1 = np.array([[1,0,0],[0,1,0],[0,0,1],[0,0,1]])
X2 = np.array([[2,3,4],[2,5,7],[0,0,3],[1,1,1]])

y1 = np.ones(4)
y2 = np.arange(4)

house_file = r"C:\Users\rimon\IML.HUJI\datasets\house_prices.csv"

def test_mse(y_true, y_pred):
    mse = mean_square_error(y_true, y_pred)
    print(mse)
    print(type(mse))
    print(sklearn_mse(y_true, y_pred))

# todo check results
def test_fit(X, y):
    print("-"*10 + "with intersect" + "-"*10)
    with_bias = LinearRegression(True)
    with_bias._fit(X,y)
    print("w shape: ",with_bias.coefs_.shape)
    print(with_bias.coefs_)
    padded_X = np.pad(X, [(0,0),(1,0)], mode="constant", constant_values=1)
    print("padded_X @ with_bias.coefs_:\n", padded_X @ with_bias.coefs_)

    print("sklearn:")
    lr_bias = LR(fit_intercept=True)
    lr_bias.fit(X,y)
    print(lr_bias.coef_)

    print("-"*10 + "without intersect" + "-"*10)
    without_bias = LinearRegression(False)
    without_bias._fit(X,y)
    print("w shape: ", without_bias.coefs_.shape)
    print(without_bias.coefs_)
    print("X @ without_bias.coefs_:\n",X@without_bias.coefs_)

    print("sklearn:")
    lr_no_bias = LR(fit_intercept=False)
    lr_no_bias.fit(X,y)
    print(lr_no_bias.coef_)

def test_loss(X,y):
    print("-"*10 + "with intersect" + "-"*10)
    with_bias = LinearRegression(True)
    with_bias._fit(X,y)#todo
    print("loss:", with_bias._loss(X,y))
    print("sklearn:")
    lr_no_bias = LR(fit_intercept=True)
    lr_no_bias.fit(X,y)
    print("loss:", lr_no_bias.loss(X,y))

    print("-"*10 + "without intersect" + "-"*10)
    without_bias = LinearRegression(False)
    without_bias._fit(X,y)
    print("loss:", without_bias._loss(X,y))

 # ======================= house price predictions ==========================

if __name__ == '__main__':
    y_true = np.array([279000, 432000, 326000, 333000, 437400, 555950])
    y_pred = np.array([199000.37562541, 452589.25533196, 345267.48129011, 345856.57131275, 563867.1347574, 395102.94362135])
    print(mean_square_error(y_true,y_pred))





