import plotly.express

import IMLearn.learners.regressors.linear_regression
from IMLearn.learners.regressors import PolynomialFitting
from IMLearn.utils import split_train_test

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio

pio.templates.default = "simple_white"

tempfile = r"C:\Users\rimon\IML.HUJI\datasets\City_Temperature.csv"
TRAIN_RATIO = 0.75


def load_data(filename: str) -> pd.DataFrame:
    """
    Load city daily temperature dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (Temp)
    """
    initial_dataset = pd.read_csv(filename,
                                  parse_dates=['Date'])
    # remove samples that have missing features
    dataset = initial_dataset.dropna(how="any")

    # add day_of_year
    dataset['DayOfYear'] = dataset['Date'].dt.dayofyear

    # remove samples with invalid features range
    dataset = dataset[dataset.Temp > -25]
    dataset = dataset[dataset.Temp < 45]


    return dataset


if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of city temperature dataset
    dataset = load_data(tempfile)

    # Question 2 - Exploring data for specific country
    # Subset the dataset to caintain samples only from the country of Israel
    israel_dataset = dataset[dataset['Country'] == 'Israel']
    # Plot a scatter plot showing `Temp`as a function of `DayOfYear`
    fig = px.scatter(israel_dataset, x='DayOfYear', y='Temp',
                     color=israel_dataset.Year.astype(str),
                     title="Average Daily Temperature "
                           "as a function of Day-Of-Year in Israel",
                     labels={'DayOfYear': "day of year",
                             'Temp': "average daily temperature",
                             'Year': "year"})
    fig.show()
    # Group the samples by `Month`,
    # and plot a bar plot showing for each month the std of the temperatures.
    temp_std_per_month = israel_dataset.groupby('Month').Temp.agg('std')
    fig = px.bar(temp_std_per_month, x=temp_std_per_month.index, y='Temp',
                 title="STD Of Daily Temperature Per Month In Israel",
                 labels={'Month': "month",
                         'Temp': "std of daily temperature"})
    fig.show()

    # Question 3 - Exploring differences between countries
    temp_average_std_per_month = dataset.groupby(['Country', 'Month'],
                                                 as_index=False).agg(
        {'Temp': ['std', 'mean']})
    temp_average_std_per_month.columns = ['Country', 'Month', 'std', 'mean']

    fig = px.line(temp_average_std_per_month, x='Month', y='mean',
                  error_y='std',
                  color=temp_average_std_per_month.Country.astype(str),
                  title="Mean Of Daily Temperature Per Month On Each Country",
                  labels={'Month': "month",
                          'mean': "mean of daily temperature"})
    fig.show()

    # Question 4 - Fitting model for different values of `k`
    # Randomly split the dataset into a training set (75%) and test set (25%).
    train_X, train_y, test_X, test_y = split_train_test(
        israel_dataset['DayOfYear'],
        israel_dataset['Temp'],
        TRAIN_RATIO)
    loss_per_k = np.zeros(10)
    # For every value k ∈ [1,10], fit a polynomial model of degree k using the training set
    for k in range(1, 11):
        poly_k = PolynomialFitting(k)
        poly_k._fit(train_X.to_numpy(), train_y.to_numpy())
        # Record the loss of the model over the test set, rounded to 2 decimal places.
        loss_per_k[k - 1] = round(
            poly_k._loss(test_X.to_numpy(), test_y.to_numpy()), 2)

    print(loss_per_k)
    loss_per_k = pd.DataFrame({"loss_per_k": loss_per_k})
    fig = px.bar(loss_per_k, x=loss_per_k.loss_per_k.index + 1, y="loss_per_k",
                 title="the test error recorded for each value of k",
                 labels={"loss_per_k": "loss per k",
                         "x": "k"})
    fig.show()

    # Question 5 - Evaluating fitted model on different countries
    chosen_k = 5
    poly_chosen = PolynomialFitting(chosen_k)
    poly_chosen._fit(israel_dataset['DayOfYear'], israel_dataset['Temp'])
    error_per_country = {"South Africa": 0, "The Netherlands": 0, "Jordan": 0}
    for country in error_per_country:
        subset = dataset[dataset['Country'] == country]
        error_per_country[country] = poly_chosen._loss(subset['DayOfYear'],
                                                       subset['Temp'])
    fig = px.bar(x=error_per_country.keys(), y=error_per_country.values(),
                 title="the model’s error over each of the other countries",
                 labels={"x": "Country", "y": "the model’s error"})
    fig.show()
