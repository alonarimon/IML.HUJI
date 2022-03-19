# todo: delete/move file?

import gaussian_estimators as ge
import numpy as np

X1 = np.arange(5)
X2 = np.arange(9)

ub = ge.UnivariateGaussian()
b = ge.UnivariateGaussian(True)


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


if __name__ == '__main__':
    # ========== UnivariateGaussian ============
    test_fit(X1)
    # test_pdf(X2)
    test_loglikelihood(2, 2, X1)
