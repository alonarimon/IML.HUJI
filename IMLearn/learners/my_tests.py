# todo: delete/move file?

import gaussian_estimators as ge
import numpy as np
from scipy.stats import multivariate_normal

X1 = np.arange(5)
X2 = np.arange(9)
Y1 = np.array([[1, 2], [1, 1], [2, 15]])
Y2 = np.random.multivariate_normal(np.array([0,15,7]), np.array([[1,0,0],[0,1,0],[0,0,1]]), size=50, check_valid='warn', tol=1e-8)

ub = ge.UnivariateGaussian()
b = ge.UnivariateGaussian(True)

multy = ge.MultivariateGaussian()

# ========== UnivariateGaussian ============

def test_fit(X):
    print("====== FIT ==========")

    print("ub.fitted_: ", ub.fitted_)
    ub.fit(X)
    print("ub.mu_: ", ub.mu_, type(ub.mu_))
    print("ub.var_: ", ub.var_, type(ub.var_))
    print("ub.fitted_: ", ub.fitted_)
    print("")

    print("b.fitted_: ", b.fitted_)
    b.fit(X)
    print("b.mu_: ", b.mu_, type(b.mu_))
    print("b.var_: ", b.var_, type(b.var_))
    print("b.fitted_: ", b.fitted_)
    print("")


def test_pdf(X):
    print("====== PDF ==========")
    print("ub.fitted_: ", ub.fitted_)
    print("ub.mu_: ", ub.mu_, type(ub.mu_))
    print("ub.var_: ", ub.var_, type(ub.var_))
    print("ub.fitted_: ", ub.fitted_)
    print("ub.pdf: ", ub.pdf(X))
    print("")

    print("b.fitted_: ", b.fitted_)
    print("b.mu_: ", b.mu_, type(b.mu_))
    print("b.var_: ", b.var_, type(b.var_))
    print("b.fitted_: ", b.fitted_)
    print("b.pdf: ", b.pdf(X))
    print("")


def test_loglikelihood(mu, sigma, X):
    print("====== loglikelihood ==========")
    ll = ge.UnivariateGaussian.log_likelihood(mu, sigma, X)
    print(ll, type(ll))

# ========== MultivariateGaussian ============

def test_multy_pdf(X):
    print("X : \n", X)
    print("fitted_: ", multy.fitted_)
    print("====== FIT ==========")
    multy.fit(X)
    print("mu_: \n", multy.mu_, multy.mu_.shape)
    print("cov_: \n", multy.cov_, multy.cov_.shape)
    print("fitted_: ", multy.fitted_)
    print("")
    print("====== PDF ==========")
    my_pdf = multy.pdf(X)
    scipy_pdf = multivariate_normal.pdf(X, mean=multy.mu_, cov=multy.cov_)
    print("multy.pdf: ", my_pdf, my_pdf.shape)
    print("scipy pdf: ", scipy_pdf)
    print("scipy result == my result: ", np.allclose(my_pdf,scipy_pdf))
    print("")

def test_multy_loglikelihood(mu, cov):
    print("====== loglikelihood ==========")
    X = np.random.multivariate_normal(mu, cov, size=50, check_valid='warn', tol=1e-8)
    ll = ge.MultivariateGaussian.log_likelihood(mu, cov, X)
    print(X)
    print(ll, type(ll))



if __name__ == '__main__':
    # ========== UnivariateGaussian ============
    # test_fit(X1)
    # # test_pdf(X2)
    # test_loglikelihood(2, 2, X1)
    # ========== MultivariateGaussian ============
    # test_multy_pdf(Y1)
    # test_multy_pdf(Y2)
    mu = np.array([0,15,7])
    cov = np.array([[1,0,0],[0,1,0],[0,0,1]])
    test_multy_loglikelihood(mu, cov)