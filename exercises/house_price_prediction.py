import os

from IMLearn.utils import split_train_test
from IMLearn.learners.regressors import LinearRegression

from typing import NoReturn
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio

pio.templates.default = "simple_white"

# features
DATE = "date"
YEAR = "year"
MONTH = "month"
DAY = "day"
ID = 'id'
PRICE = 'price'
ZIP = 'zipcode'

RESPONSE_COLUMN = 'price'
FEATURES_TO_DROP = ['id', 'date', 'yr_renovated', 'zipcode']  # we want to drop the house ID
NUM_AREAS = 20

HOUSES_FILE = r"IML.HUJI\datasets\house_prices.csv"
EVALUATION_OUTPUT_PATH = r"C:\Users\rimon\Desktop\2021-2022 semester B\IML\ex2\feature_evaluation_graph"
TRAIN_RATIO = 0.75

#Q4
Q4_OUTPUT_PATH = r"C:\Users\rimon\Desktop\2021-2022 semester B\IML\ex2\a "
Q4_GRAPH_TITLE = "average loss as a function of training size"
Q4_X_TITLE = "training-data size"
Q4_Y_TITLE = "average loss"

def load_data(filename: str):
    """
    Load house prices dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (prices) - either as a single
    DataFrame or a Tuple[DataFrame, Series]
    """
    initial_dataset = pd.read_csv(filename)
    samples_mat, response = preprocess_dataset(initial_dataset, RESPONSE_COLUMN)
    return (samples_mat, response)


def preprocess_dataset(initial_dataset: pd.DataFrame,
                       response_label) -> (pd.DataFrame, pd.Series):
    """
    preprocessing the data set to include only
    the relevance features and samples, and won't include the response
    :param response_label: the name of the column of the response.
            this column will be removed from the features of the dataset
    :param initial_dataset: the initial dataset
    :return: samples matrix after preprocess
    """
    # remove samples that have missing features
    samples_dataset = initial_dataset.dropna(how="any")
    # remove duplicated samples
    samples_dataset = samples_dataset.drop_duplicates()

    # remove samples with invalid features range
    samples_dataset = samples_dataset[samples_dataset.price > 0]
    samples_dataset = samples_dataset[samples_dataset.bedrooms < 15]

    # split the date feature
    date = samples_dataset["date"]
    samples_dataset["year_sold"], samples_dataset["month_sold"], samples_dataset["day_sold"] = \
        date.str[0:4].astype(int), date.str[4:6].astype(int), date.str[6:8].astype(int)

    # normalize the above and basement features
    samples_dataset['sqft_above'] = samples_dataset['sqft_above'] / \
                                    samples_dataset['sqft_living']
    samples_dataset['sqft_basement'] = samples_dataset['sqft_basement'] / \
                                    samples_dataset['sqft_living']

    # add years since last renovated- where the yr_built counts as reanovation year.
    last_renovation = samples_dataset[['yr_built', 'yr_renovated']].max(axis=1)
    years_since_renovated = samples_dataset['year_sold'] - last_renovation
    samples_dataset['yrs_since_renovated'] = years_since_renovated

    # quantization of the zipcodes to area-groups
    samples_dataset['zipcode_area'] = pd.qcut(samples_dataset['zipcode'], NUM_AREAS, labels=False)
    samples_dataset = pd.get_dummies(samples_dataset, columns=['zipcode_area'])

    # always drop the response information
    response = samples_dataset[response_label]
    samples_dataset = samples_dataset.drop(columns=response_label)
    # drop the un-relevant information
    samples_dataset = samples_dataset.drop(columns=FEATURES_TO_DROP)

    return samples_dataset, response


def feature_evaluation(X: pd.DataFrame, y: pd.Series,
                       output_path: str = ".") -> NoReturn:
    """
    Create scatter plot between each feature and the response.
        - Plot title specifies feature name
        - Plot title specifies Pearson Correlation between feature and response
        - Plot saved under given folder with file name including feature name
    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Design matrix of regression problem

    y : array-like of shape (n_samples, )
        Response vector to evaluate against

    output_path: str (default ".")
        Path to folder in which plots are saved
    """
    for feature in X.columns:
        plot_feature_response_correlation(X[feature], y, output_path)

def pearson_correlation(X, Y):
    return np.cov(X, Y)[0, 1] / (np.std(X) * np.std(Y))

def plot_feature_response_correlation(feature: pd.Series, response: pd.Series, output_path):
    p_c = pearson_correlation(feature, response)
    feature_name = str(feature.name)
    title = f"{feature_name} and {RESPONSE_COLUMN} correlation\npearson correlation: {p_c}"
    file_path = os.path.join(output_path, feature_name+'.png')
    data = go.Scatter(x=feature, y=response, mode='markers', marker=dict(size=2))
    fig = go.Figure(data=data) \
        .update_layout(title_text=title) \
        .update_xaxes(title_text=(str(feature_name)+" value")) \
        .update_yaxes(title_text="response value")
    pio.write_image(fig, file_path, format='png')

if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of housing prices dataset
    X, y = load_data(HOUSES_FILE)

    # Question 2 - Feature evaluation with respect to response
    feature_evaluation(X, y, EVALUATION_OUTPUT_PATH)

    # Question 3 - Split samples into training- and testing sets.
    train_X, train_y, test_X, test_y = split_train_test(X, y, TRAIN_RATIO)

    # Question 4 - Fit model over increasing percentages of the overall training data
    p_range = np.arange(0.1, 1.01, 0.01)
    mean_losses = np.zeros(p_range.shape)
    variance_losses = np.zeros(p_range.shape)
    linear_model = LinearRegression(True)
    # For every percentage p in 10%, 11%, ..., 100%, repeat the following 10 times:
    for p in range(p_range.shape[0]):
        p_losses = np.zeros(10)
        for i in range(10):
    #   1) Sample p% of the overall training data
            p_train_samples = train_X.sample(frac=p_range[p])
            p_train_response = train_y[p_train_samples.index]
    #   2) Fit linear model (including intercept) over sampled set
            linear_model._fit(p_train_samples.to_numpy(), p_train_response.to_numpy())
    #   3) Test fitted model over test set
            p_loss = linear_model._loss(test_X.to_numpy(), test_y.to_numpy())
    #   4) Store average and variance of loss over test set
            p_losses[i] = p_loss
        mean_losses[p] = p_losses.mean()
        variance_losses[p] = p_losses.var()

    x_axis = p_range * train_X.shape[0]

    # Then plot average loss as function of training size with error ribbon of size (mean-2*std, mean+2*std)
    data = go.Scatter(x=x_axis, y=mean_losses, mode="markers+lines", name="Mean Losses", line=dict(dash="dash"), marker=dict(color="green", opacity=.7))

    # compute error ribbon:
    std_losses = np.power(variance_losses,0.5)
    down_ribbon = go.Scatter(x=x_axis, y=mean_losses-2*(std_losses), fill=None, mode="lines", line=dict(color="lightgrey"), showlegend=False)
    up_ribbon = go.Scatter(x=x_axis, y=mean_losses+2*(std_losses), fill='tonexty', mode="lines", line=dict(color="lightgrey"), showlegend=False)

    fig = go.Figure(data=(data, down_ribbon, up_ribbon)) \
        .update_layout(title_text=Q4_GRAPH_TITLE) \
        .update_xaxes(title_text=Q4_X_TITLE) \
        .update_yaxes(title_text=Q4_Y_TITLE)

    pio.write_image(fig, Q4_OUTPUT_PATH, format='png')
