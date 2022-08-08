# REGRESSION MODULE


#! IMPORTS


from typing import Union
from scipy.linalg import pinv
import numpy as np
import pandas as pd


#! CLASSES


class LinearRegression:
    """
    Obtain the regression coefficients according to the Ordinary Least
    Squares approach.

    Parameters
    ----------
    y:  (samples, dimensions) numpy array or pandas.DataFrame
        the array containing the dependent variable.

    x:  (samples, features) numpy array or pandas.DataFrame
        the array containing the indipendent variables.

    fit_intercept: bool
        Should the intercept be included in the model?
        Otherwise it will be set to zero.
    """

    def __init__(
        self,
        y: Union[np.ndarray, pd.DataFrame],
        x: Union[np.ndarray, pd.DataFrame],
        fit_intercept: bool = True,
    ):

        # add the input parameters
        txt = "{} must be a {} object.".format("fit_intercept", "bool")
        assert isinstance(fit_intercept, bool), txt
        self.fit_intercept = fit_intercept

        # correct the shape of y and x
        self.Y = self._simplify(y)
        self.X = self._simplify(x)
        txt = "'X' and 'Y' number of rows must be identical."
        assert self.X.shape[0] == self.Y.shape[0], txt

        # add the ones for the intercept
        if self.fit_intercept:
            xx = self.X.values
            xx = np.hstack([np.ones((self.X.shape[0], 1)), xx])

        # get the coefficients and intercept
        self.betas = pd.DataFrame(
            data=pinv(xx.T.dot(xx)).dot(xx.T).dot(self.Y.values),
            index=["INTERCEPT"] + self.X.columns.to_numpy().tolist(),
            columns=self.Y.columns.to_numpy().tolist(),
        )

    def _simplify(self, v):
        """
        internal method to check entries in the constructor.
        """
        txt = "the input object must be a pandas.DataFrame, a "
        txt += "numpy.ndarray or None."
        if isinstance(v, pd.DataFrame):
            xx = v.astype(float)
        elif isinstance(v, np.ndarray):
            xx = pd.DataFrame(np.atleast_1d(v).astype(float))
        else:
            raise NotImplementedError(txt)
        return xx

    def __call__(self, x):
        """
        predict the fitted Y value according to the provided x.
        """
        return self._predict(self._simplify(x))

    def _predict(self, xx):
        """
        predict the fitted Y value according to the provided x.
        """
        if self.fit_intercept:
            b0 = self.betas.iloc[0]
            bs = self.betas.iloc[1:]
            zz = xx.dot(bs) + b0
        else:
            zz = xx.dot(self.betas)
        return zz


class PolynomialRegression(LinearRegression):
    """
    Obtain the regression coefficients according to the Ordinary Least
    Squares approach.

    Parameters
    ----------
    y:  (samples, dimensions) numpy array or pandas.DataFrame
        the array containing the dependent variable.

    x:  (samples, 1) numpy array or pandas.DataFrame
        the array containing the indipendent variables.

    n: int
        the order of the polynome.

    fit_intercept: bool
        Should the intercept be included in the model?
        Otherwise it will be set to zero.
    """

    def __init__(
        self,
        y: Union[np.ndarray, pd.DataFrame],
        x: Union[np.ndarray, pd.DataFrame],
        n: int = 1,
        fit_intercept: bool = True,
    ):
        # add the input parameters
        txt = "{} must be a {} object.".format("fit_intercept", "bool")
        assert isinstance(fit_intercept, bool), txt
        self.fit_intercept = fit_intercept
        assert isinstance(n, int), txt.format("n", "int")
        self.n = n

        # correct the shape of y and x
        self.Y = self._simplify(y)
        self.X = self._simplify(x, self.n)
        txt = "'X' and 'Y' number of rows must be identical."
        assert self.X.shape[0] == self.Y.shape[0], txt

        # add the ones for the intercept
        if self.fit_intercept:
            xx = self.X.values
            xx = np.hstack([np.ones((self.X.shape[0], 1)), xx])

        # get the coefficients and intercept
        self.betas = pd.DataFrame(
            data=pinv(xx.T.dot(xx)).dot(xx.T).dot(self.Y.values),
            index=["INTERCEPT"] + self.X.columns.to_numpy().tolist(),
            columns=self.Y.columns.to_numpy().tolist(),
        )

    def _simplify(self, v, n=None):
        """
        internal method to check entries in the constructor.
        """
        txt = "the input object must be a pandas.DataFrame, a numpy.ndarray"
        txt += " or None."
        if isinstance(v, pd.DataFrame):
            xx = v.astype(float)
        elif isinstance(v, np.ndarray):
            xx = pd.DataFrame(np.atleast_1d(v).astype(float))
        else:
            raise NotImplementedError(txt)
        if n is not None:
            xx = pd.concat([xx ** (i + 1) for i in range(n)], axis=1)
            cols = ["C{}".format(i + 1) for i in range(xx.shape[1])]
            xx.columns = pd.Index(cols)
        return xx

    def __call__(self, x):
        """
        predict the fitted Y value according to the provided x.
        """
        return self._predict(self._simplify(x, self.n))


class PowerRegression(LinearRegression):
    """
    Obtain the regression coefficients according to the power model:

                y = b_0 * x_1 ^ b_1 * ... * x_n ^ b_n

    Input

        y (2D column numpy array)

            the array containing the dependent variable.

        x (2D column numpy array)

            the array containing the indipendent variable.
            The number of rows must equal the rows of y, while
            each column will be a regressor (i.e. an indipendent
            variable).

        digits (int)

            the number of digits used to print the coefficients
    """

    def __init__(self, y, x):
        """
        constructor
        """
        super().__init__(y=self._simplify(y), x=self._simplify(x))
        self.betas.loc[0] = np.e ** self.betas.loc[0]

    def __call__(self, x):
        """
        predict the fitted Y value according to the provided x.
        """
        X = self._simplify(x)
        m = self.betas.shape[0] - 1
        assert X.shape[1] == m, "'X' must have {} columns.".format(m)
        Z = []
        for dim in self.betas:
            coefs = self.betas[dim].values.T
            Z += [np.prod(X ** coefs[1:], axis=1) * coefs[0]]
        Z = pd.DataFrame(np.atleast_2d(Z).T)
        if isinstance(x, pd.DataFrame):
            idx = x.index
        else:
            idx = pd.Index(np.arange(X.shape[0]))
        return pd.DataFrame(Z, index=idx, columns=self.__dv_labels__)


class HyperbolicRegression(LinearRegression):
    """
    Obtain the regression coefficients according to the (Rectangular) Least
    Squares Hyperbolic function:
                                                        b * x
                    (x + a) * (y + b) = a * b ==> y = - -----
                                                        a + x
    Input

        y (2D column numpy array)

            the array containing the dependent variable.

        x (2D column numpy array)

            the array containing the indipendent variable.
            The number of rows must equal the rows of y, while
            each column will be a regressor (i.e. an indipendent
            variable).

        digits (int)

            the number of digits used to render the coefficients.
    """

    def __init__(self, y, x):
        """
        constructor
        """
        super().__init__(y=y, x=x, fit_intercept=True)

        # obtain the hyberbolic coefficients
        # a = -1 / intercept
        # b =  slope / intercept
        self.a = -1 / self.betas.values.flatten()[0]
        self.b = self.betas.values.flatten()[1] / self.betas.values.flatten()[0]

    def __call__(self, x):
        """
        predict the fitted Y value according to the provided x.
        """
        X = self._simplify(x)
        assert X.shape[1] == 1, "'X' must have 1 column."
        Z = -self.b * X / (self.a + X)
        if isinstance(x, pd.DataFrame):
            idx = x.index
        else:
            idx = pd.Index(np.arange(X.shape[0]))
        return pd.DataFrame(Z, index=idx, columns=self.coefs.columns)


# test the classes
if __name__ == "__main__":

    # LINEAR REGRESSION
    x = np.arange(10)
    y = x * 1.5 + 0.5 + np.random.randn(len(x)) * 0.25
    print("betas = " + str(LinearRegression(y=y, x=x).betas))
    print(LinearRegression(y=y, x=x)(x))
