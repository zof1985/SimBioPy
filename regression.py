# REGRESSION MODULE


#! IMPORTS


from typing import Union
import numpy as np
import pandas as pd
import scipy.linalg as sl
import sympy as sy


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
            data=sl.pinv(xx.T.dot(xx)).dot(xx.T).dot(self.Y.values),
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

    def predict(self, x):
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
            data=sl.pinv(xx.T.dot(xx)).dot(xx.T).dot(self.Y.values),
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

    def predict(self, x):
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

    def __init__(self, y, x, digits=5):
        """
        constructor
        """

        # correct the shape of y and x
        assert isinstance(digits, (int)), "'digits' must be and 'int'."
        self.digits = digits
        YY = np.log(self.__simplify__(y, "Y", None))
        XX = np.log(self.__simplify__(x, "X", None))
        txt = "'X' and 'Y' number of rows must be identical."
        assert XX.shape[0] == YY.shape[0], txt

        # add the ones for the intercept
        XX = np.hstack([np.ones((XX.shape[0], 1)), XX])

        # get the coefficients
        coefs = sl.pinv(XX.T.dot(XX)).dot(XX.T).dot(YY)
        coefs[0] = np.e ** coefs[0]

        # get the coefficients and intercept
        self._coefs = pd.DataFrame(
            data=coefs, index=self.__iv_labels__, columns=self.__dv_labels__
        )

        # obtain the symbolic representation of the equation
        self.symbolic = []
        for c, var in enumerate(self.betas):
            vrs = self.X.columns.to_numpy()
            a = self.betas[var].values[0]
            bs = self.betas[var].values[1:]
            line = sy.Float(a, digits)
            for v, b in zip(vrs, bs):
                line = line * sy.symbols(v) ** sy.Float(b, digits)
            self.symbolic += [sy.Eq(sy.symbols(var), line)]

    def copy(self):
        """
        copy the current instance
        """
        return PowerRegression(self.Y, self.X)

    def predict(self, x):
        """
        predict the fitted Y value according to the provided x.
        """
        X = self.__simplify__(x, "x", None)
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

    @property
    def __iv_labels__(self):
        """
        return the labels for the regressors.
        """
        lbls = ["b{}".format(i + 1) for i in np.arange(len(self.X.columns))]
        return ["b0"] + lbls


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

    def __init__(self, y, x, digits=5):
        """
        constructor
        """

        # add the input parameters
        txt = "{} must be a {} object."
        assert isinstance(digits, (int)), txt.format("digits", "int")
        self.digits = digits

        # correct the shape of y and x and get their reciprocal values
        YY = self.__simplify__(y, "Y", None) ** (-1)
        XX = self.__simplify__(x, "X", None) ** (-1)
        txt = "'X' and 'Y' number of rows must be identical."
        assert XX.shape[0] == YY.shape[0], txt
        assert XX.shape[1] == 1, "'X' must have just 1 column."

        # add the ones for the intercept
        XX = np.hstack([np.ones((XX.shape[0], 1)), XX])

        # get the linear regression coefficients
        coefs = sl.pinv(XX.T.dot(XX)).dot(XX.T).dot(YY)

        # obtain the hyberbolic coefficients
        # a = -1 / intercept
        # b =  slope / intercept
        coefs = [[coefs[1][0] / coefs[0][0]], [-1 / coefs[0][0]]]
        self._coefs = pd.DataFrame(
            data=coefs, index=self.__iv_labels__, columns=self.__dv_labels__
        )

        # obtain the symbolic representation of the equation
        x = sy.symbols(self.X.columns.to_numpy()[0])
        c = [sy.Float(i, self.digits) for i in self.betas.values]
        y = sy.symbols(self.betas.columns.to_numpy()[0])
        self.symbolic = [sy.Eq(y, (c[1] * x) / (c[0] + x))]

    def copy(self):
        """
        copy the current instance
        """
        return HyperbolicRegression(self.Y, self.X)

    def predict(self, x):
        """
        predict the fitted Y value according to the provided x.
        """
        X = self.__simplify__(x, "x", None)
        assert X.shape[1] == 1, "'X' must have 1 column."
        Z = -self.betas.loc["b"].values * X / (self.betas.loc["a"].values + X)
        if isinstance(x, pd.DataFrame):
            idx = x.index
        else:
            idx = pd.Index(np.arange(X.shape[0]))
        return pd.DataFrame(Z, index=idx, columns=self.__dv_labels__)

    @property
    def __iv_labels__(self):
        """
        return the labels for the regressors.
        """
        return ["a", "b"]

    def to_string(self, digits=3):
        """
        Create a textual representation of the fitted model.

        Input:
            digits: (int)
                    the number of digits to be used to represent the
                    coefficients.

        Output:
            txt:    (str)
                    a single string representing the fitted model.
        """
        frm = "{:+." + str(digits) + "f}"
        txt = "({} " + frm + ") * ({} " + frm + ") = " + frm
        return txt.format(
            self.betas.columns.to_numpy()[0],
            self.betas.loc["a"].values.flatten()[0],
            self.__dv_labels__()[0],
            self.betas.loc["b"].values.flatten()[0],
            np.prod(self.betas.values.flatten()),
        )
