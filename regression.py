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
    ) -> None:

        # set the inputs
        self._set_inputs(y=y, x=x, fit_intercept=fit_intercept)

        # calculate betas
        self._calculate_betas()

    def _simplify(
        self,
        v: Union[np.ndarray, pd.DataFrame],
        label: str = "",
    ) -> pd.DataFrame:
        """
        internal method to format the entries in the constructor and call
        methods.

        Parameters
        ----------
        v: np.ndarray | pd.DataFrame
            the data to be formatter

        label: str
            in case an array is provided, the label is used to define the
            columns of the output DataFrame.

        Returns
        -------
        d: pd.DataFrame
            the data formatted as DataFrame.
        """
        if isinstance(v, pd.DataFrame):
            return v.astype(float)
        if isinstance(v, np.ndarray):
            if v.ndim == 1:
                d = np.atleast_2d(v).T
            elif v.ndim == 2:
                d = v
            else:
                raise ValueError(v)
            cols = ["{}{}".format(label, i) for i in range(d.shape[1])]
            return pd.DataFrame(d.astype(float), columns=cols)
        raise NotImplementedError(v)

    def _add_intercept(self, x: pd.DataFrame) -> None:
        """
        add an intercept to x.
        """
        x.insert(0, "INTERCEPT", np.tile(1, x.shape[0]))

    def _set_inputs(
        self,
        y: Union[np.ndarray, pd.DataFrame],
        x: Union[np.ndarray, pd.DataFrame],
        fit_intercept: bool = True,
    ) -> None:
        """
        set the input parameters
        """
        # add the input parameters
        txt = "{} must be a {} object.".format("fit_intercept", "bool")
        assert isinstance(fit_intercept, bool), txt
        self.fit_intercept = fit_intercept

        # correct the shape of y and x
        self.y = self._simplify(y, "Y")
        self.x = self._simplify(x, "X")
        txt = "'x' and 'y' number of rows must be identical."
        assert self.x.shape[0] == self.y.shape[0], txt

    def _calculate_betas(self) -> None:
        """
        calculate the beta coefficients.
        """

        # add the ones for the intercept
        betas = self.x.copy()
        if self.fit_intercept:
            self._add_intercept(betas)

        # get the coefficients and intercept
        self.betas = (pinv((betas.T @ betas).values) @ betas.T) @ self.y
        labels = ["beta{}".format(i) for i in range(self.betas.shape[0])]
        self.betas.index = pd.Index(labels)

    def __repr__(self) -> str:
        """
        representation of the object.
        """
        return self.betas.__repr__()

    def __str__(self) -> str:
        """
        representation of the object.
        """
        return self.betas.__str__()

    def __call__(self, x: Union[np.ndarray, pd.DataFrame]) -> pd.DataFrame:
        """
        predict the fitted Y value according to the provided x.

        Parameters
        ----------
        x: np.ndarray | pd.DataFrame
            the input data used as predictor

        Returns
        -------
        y: pd.DataFrame
            the predicted values
        """
        v = self._simplify(x, "X")
        if self.fit_intercept:
            self._add_intercept(v)
        return v.values @ self.betas


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
    ) -> None:
        self._set_inputs(y=y, x=x, fit_intercept=fit_intercept)

        # set the polynomial order
        assert isinstance(n, int), ValueError(n)
        assert n > 0, "'n' must be > 0"
        self.n = n

        # get the coefficients
        self._calculate_betas()

    def _expand_to_n(self, df) -> pd.DataFrame:
        """
        expand the df values up to the n-th order.
        """
        betas = []
        for i in range(self.n):
            b_new = df.copy()
            cols = [j + "{}".format(i + 1) for j in b_new.columns]
            b_new.columns = pd.Index(cols)
            betas += [b_new ** (i + 1)]
        return pd.concat(betas, axis=1)

    def _calculate_betas(self) -> None:
        """
        calculate the beta coefficients.
        """
        # expand x to cope with the polynomial order
        betas = self._expand_to_n(self.x)

        # add the ones for the intercept
        if self.fit_intercept:
            self._add_intercept(betas)

        # get the coefficients and intercept
        self.betas = (pinv((betas.T @ betas).values) @ betas.T) @ self.y
        labels = ["beta{}".format(i) for i in range(self.betas.shape[0])]
        self.betas.index = pd.Index(labels)

    def __call__(self, x: Union[np.ndarray, pd.DataFrame]) -> pd.DataFrame:
        """
        predict the fitted Y value according to the provided x.

        Parameters
        ----------
        x: np.ndarray | pd.DataFrame
            the input data used as predictor

        Returns
        -------
        y: pd.DataFrame
            the predicted values
        """
        v = self._expand_to_n(self._simplify(x, "X"))
        if self.fit_intercept:
            self._add_intercept(v)
        return v.values @ self.betas


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
    y = x * 1.5 + 0.5 + np.random.randn(len(x)) * 0.01
    lr = LinearRegression(y=y, x=x)
    print(lr)
    print(lr(x))

    # POLYNOMIAL REGRESSION
    pr = PolynomialRegression(y=y, x=x, n=2)
    print(pr)
    print(pr(x))
