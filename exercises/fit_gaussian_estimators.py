from IMLearn.learners import UnivariateGaussian, MultivariateGaussian
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
pio.templates.default = "simple_white"

# EX1 Q3.1
MU_Q1 = 10
VAR_Q1 = 1
SAMPLES_NUM_Q1 = 1000
JUMP_Q1 = 10

# EX1 Q3.2
MU_Q2 = np.array([0, 0, 4, 0])
COV_Q2 = np.array([[1, 0.2, 0, 0.5], [0.2, 2, 0, 0], [0, 0, 1, 0], [0.5, 0, 0, 1]])
SAMPLES_NUM_Q2 = 1000
LINE_SIZE_Q2 = 200


def test_univariate_gaussian():
    # todo: add all output to the PDF
    # Question 1 - Draw samples and print fitted model
    unbiased_g = UnivariateGaussian()
    samples = np.random.normal(MU_Q1, VAR_Q1, SAMPLES_NUM_Q1)
    unbiased_g.fit(samples)

    print(f"({unbiased_g.mu_}, {unbiased_g.var_})")

    # Question 2 - Empirically showing sample mean is consistent
    samples_sizes = np.arange(JUMP_Q1, SAMPLES_NUM_Q1+JUMP_Q1, JUMP_Q1)
    distance = np.zeros(samples_sizes.shape[0])
    for i in range(samples_sizes.shape[0]):  # todo: check if we can do without loop
        unbiased_g.fit(samples[:samples_sizes[i]])
        distance[i] = abs(MU_Q1 - unbiased_g.mu_)

    # plotting results
    data = go.Scatter(x=samples_sizes, y=distance, mode='markers+lines')
    fig = go.Figure(data=data)\
        .update_layout(title_text="Expectation estimation error per sample-size")\
        .update_xaxes(title_text="Sample size")\
        .update_yaxes(title_text="Estimation error")
    fig.show()

    # Question 3 - Plotting Empirical PDF of fitted model
    pdfs = unbiased_g.pdf(samples) #todo: add to the PDF + What are you expecting to see in the plot?
    data = go.Scatter(x=samples, y=pdfs, mode='markers')
    fig = go.Figure(data=data) \
        .update_layout(title_text="Empirical PDF of fitted model") \
        .update_xaxes(title_text="Samples") \
        .update_yaxes(title_text="Empirical PDF value")
    fig.show()


def test_multivariate_gaussian():
    # Question 4 - Draw samples and print fitted model
    multy = MultivariateGaussian()
    samples = np.random.multivariate_normal(MU_Q2, COV_Q2, SAMPLES_NUM_Q2)
    multy.fit(samples)

    print(multy.mu_)
    print(multy.cov_)


    # Question 5 - Likelihood evaluation
    f_values = np.linspace(-10, 10, LINE_SIZE_Q2)
    f1_f2_values = np.array(np.meshgrid(f_values, f_values))
    z = np.zeros((LINE_SIZE_Q2, LINE_SIZE_Q2))
    expectations = np.array([f1_f2_values[0, :, :], z, f1_f2_values[1, :, :], z]).T

    ll = np.apply_along_axis(MultivariateGaussian.log_likelihood, 2, expectations, COV_Q2, samples)

    heatmap = go.Heatmap(x=f_values, y=f_values, z=ll)
    layout = go.Layout(title="log-likelihood calculated per mu values")
    fig = go.Figure(data=heatmap, layout=layout).update_xaxes(title_text="f3 value") \
        .update_yaxes(title_text="f1 value")
    fig.show()


    # Question 6 - Maximum likelihood
    # raise NotImplementedError()


if __name__ == '__main__':
    np.random.seed(0)
    # test_univariate_gaussian()
    test_multivariate_gaussian()
