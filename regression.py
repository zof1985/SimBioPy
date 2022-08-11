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
        y: Union[np.ndarray, pd.DataFrame, list, int, float],
        x: Union[np.ndarray, pd.DataFrame, list, int, float],
        fit_intercept: bool = True,
    ) -> None:

        # set the inputs
        self._set_inputs(y=y, x=x, fit_intercept=fit_intercept)

        # calculate betas
        self._calculate_betas()

    def _simplify(
        self,
        v: Union[np.ndarray, pd.DataFrame, list, int, float],
        label: str = "",
    ) -> pd.DataFrame:
        """
        internal method to format the entries in the constructor and call
        methods.

        Parameters
        ----------
        v: np.ndarray | pd.DataFrame | list | int | float
            the data to be formatter

        label: str
            in case an array is provided, the label is used to define the
            columns of the output DataFrame.

        Returns
        -------
        d: pd.DataFrame
            the data formatted as DataFrame.
        """

        def simplify_array(v: np.ndarray, l: str) -> pd.DataFrame:
            if v.ndim == 1:
                d = np.atleast_2d(v).T
            elif v.ndim == 2:
                d = v
            else:
                raise ValueError(v)
            cols = [f"{l}{i}" for i in range(d.shape[1])]
            return pd.DataFrame(d.astype(float), columns=cols)

        if isinstance(v, pd.DataFrame):
            return v.astype(float)
        if isinstance(v, list):
            return simplify_array(np.array(v), label)
        if isinstance(v, np.ndarray):
            return simplify_array(v, label)
        if np.isreal(v):
            return simplify_array(np.array([v]), label)
        raise NotImplementedError(v)

    def _add_intercept(
        self,
        x: pd.DataFrame,
    ) -> None:
        """
        add an intercept to x.
        """
        x.insert(0, "INTERCEPT", np.tile(1, x.shape[0]))

    def _set_inputs(
        self,
        y: Union[np.ndarray, pd.DataFrame, list, int, float],
        x: Union[np.ndarray, pd.DataFrame, list, int, float],
        fit_intercept: bool = True,
    ) -> None:
        """
        set the input parameters
        """
        # add the input parameters
        assert isinstance(fit_intercept, bool), ValueError(fit_intercept)
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
        labels = self.betas.shape[0]
        labels = [i + (0 if self.fit_intercept else 1) for i in range(labels)]
        self.betas.index = pd.Index([f"beta{i}" for i in labels])

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

    def __call__(
        self,
        x: Union[np.ndarray, pd.DataFrame, list, int, float],
    ) -> pd.DataFrame:
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
        y: Union[np.ndarray, pd.DataFrame, list, int, float],
        x: Union[np.ndarray, pd.DataFrame, list, int, float],
        n: int = 1,
        fit_intercept: bool = True,
    ) -> None:

        # set the polynomial order
        assert isinstance(n, int), ValueError(n)
        assert n > 0, "'n' must be > 0"
        self.n = n
        super().__init__(y=y, x=x, fit_intercept=fit_intercept)

    def _expand_to_n(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        expand the df values up to the n-th order.
        """
        betas = []
        for i in range(self.n):
            b_new = df.copy()
            cols = [j + f"{i + 1}" for j in b_new.columns]
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
        labels = self.betas.shape[0]
        labels = [i + (0 if self.fit_intercept else 1) for i in range(labels)]
        self.betas.index = pd.Index([f"beta{i}" for i in labels])

    def __call__(
        self,
        x: Union[np.ndarray, pd.DataFrame, list, int, float],
    ) -> pd.DataFrame:
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

    Parameters
    ----------
    y:  (samples, dimensions) numpy array or pandas.DataFrame
        the array containing the dependent variable.

    x:  (samples, features) numpy array or pandas.DataFrame
        the array containing the indipendent variables.
    """

    def __init__(
        self,
        y: Union[np.ndarray, pd.DataFrame, list, int, float],
        x: Union[np.ndarray, pd.DataFrame, list, int, float],
    ) -> None:
        super().__init__(y=y, x=x, fit_intercept=True)

    def _set_inputs(
        self,
        y: Union[np.ndarray, pd.DataFrame, list, int, float],
        x: Union[np.ndarray, pd.DataFrame, list, int, float],
        fit_intercept: bool = True,
    ) -> None:
        """
        set the input parameters
        """
        super()._set_inputs(y=y, x=x, fit_intercept=fit_intercept)

        # check both x and y are positive
        assert np.all(self.y.values > 0), "'y' must be positive only."
        assert np.all(self.x.values > 0), "'x' must be positive only."

    def _calculate_betas(self) -> None:
        """
        calculate the beta coefficients.
        """
        # add the ones for the intercept
        betas = self.x.applymap(np.log)
        self._add_intercept(betas)

        # get the coefficients and intercept
        logy = self.y.applymap(np.log)
        self.betas = (pinv((betas.T @ betas).values) @ betas.T) @ logy
        self.betas.iloc[0] = np.e ** self.betas.iloc[0]
        labels = [i for i in range(self.betas.shape[0])]
        self.betas.index = pd.Index([f"beta{i}" for i in labels])

    def __call__(
        self,
        x: Union[np.ndarray, pd.DataFrame, list, int, float],
    ) -> pd.DataFrame:
        """
        predict the fitted Y value according to the provided x.
        """
        v = self._simplify(x, "X")
        o = np.ones((v.shape[0], self.betas.shape[1]))
        o = pd.DataFrame(o, index=v.index, columns=self.betas.columns)
        o *= self.betas.iloc[0].values
        for i in np.arange(1, self.betas.shape[0]):
            o *= v.values ** self.betas.iloc[i].values
        return o


class HyperbolicRegression(PowerRegression):
    """
    Obtain the regression coefficients according to the (Rectangular) Least
    Squares Hyperbolic function:
                            y = eiv_pos / x + b
    Parameters
    ----------
    y:  (samples, dimensions) numpy array or pandas.DataFrame
        the array containing the dependent variable.

    x:  (samples, features) numpy array or pandas.DataFrame
        the array containing the indipendent variables.
    """

    def __init__(
        self,
        y: Union[np.ndarray, pd.DataFrame, list, int, float],
        x: Union[np.ndarray, pd.DataFrame, list, int, float],
    ) -> None:
        super().__init__(y=y, x=x)

    def _calculate_betas(self) -> None:
        """
        calculate the beta coefficients.
        """
        betas = self.x ** (-1)
        self._add_intercept(betas)
        self.betas = (pinv((betas.T @ betas).values) @ betas.T) @ self.y
        labels = [i for i in range(self.betas.shape[0])]
        self.betas.index = pd.Index([f"beta{i}" for i in labels])

    def __call__(
        self,
        x: Union[np.ndarray, pd.DataFrame, list, int, float],
    ) -> pd.DataFrame:
        """
        predict the fitted Y value according to the provided x.
        """
        v = self._simplify(x, "X") ** (-1)
        self._add_intercept(v)
        return v.values @ self.betas


class _Axis(LinearRegression):
    """
    generate the axis object defining one single axis of a 2D geometric figure.

    Parameters:
    y:  (samples, dimensions) numpy array or pandas.DataFrame
        the array containing the dependent variable.

    x:  (samples, features) numpy array or pandas.DataFrame
        the array containing the indipendent variables.
    """
    _vertex = (None, None)

    def __init__(
        self,
        y: Union[np.ndarray, pd.DataFrame, list, int, float],
        x: Union[np.ndarray, pd.DataFrame, list, int, float],
    ) -> None:
        super().__init__(y=y, x=x, fit_intercept=True)
        txt = "Axis must be defined by 2 elements only."
        assert self.y.shape[0] == 2, txt
        assert self.x.shape[0] == 2, txt

        # set the vertex
        x = self.x.values.flatten()
        y = self.y.values.flatten()
        self._vertex = ((x[0], y[0]), (x[1], y[1]))

    @property
    def angle(self) -> float:
        """
        return the angle (in radians) of the axis.
        """
        return np.arctan(self.betas.loc["beta1"].values[0][0])

    @property
    def length(self) -> float:
        """
        get the distance between the two vertex.
        """
        a, b = self._vertex
        return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5

    @property
    def vertex(self) -> tuple:
        """
        return the vertex of the axis
        """
        return self._vertex


class EllipsisRegression(LinearRegression):
    """
    calculate the beta coefficients equivalent to the fit the coefficients
    eiv_pos,b,cnd,d,e,f, representing an ellipse described by the formula

        b_0 * x^2 + b_1 * xy + b_2 * y^2 + b_3 * x + b_4 * y + b_5 = 0

    Based on the algorithm of Halir and Flusser.

    References
    ----------
    Halır R, Flusser J. Numerically stable direct least squares fitting of
        ellipses. InProc. 6th International Conference in Central Europe on
        Computer Graphics and Visualization. WSCG 1998 (Vol. 98, pp. 125-132).
        Citeseer. https://citeseerx.ist.psu.edu/viewdoc/download;jsessionid=DF7A4B034A45C75AFCFF861DA1D7B5CD?doi=10.1.1.1.7559&rep=rep1&type=pdf

    Parameters
    ----------
    y:  (samples, dimensions) numpy array or pandas.DataFrame
        the array containing the dependent variable.

    x:  (samples, features) numpy array or pandas.DataFrame
        the array containing the indipendent variables.
    """

    def __init__(
        self,
        y: Union[np.ndarray, pd.DataFrame, list, int, float],
        x: Union[np.ndarray, pd.DataFrame, list, int, float],
    ) -> None:
        super().__init__(y=y, x=x)
        assert self.x.shape[1] == 1, "x can be unidimensional only"
        assert self.y.shape[1] == 1, "y can be unidimensional only"

    def _calculate_betas(self) -> None:
        """
        calculate the regression coefficients.
        """
        # quadratic part of the design matrix
        xval = self.x.values.flatten()
        yval = self.y.values.flatten()
        d_1 = np.vstack([xval**2, xval * yval, yval**2]).T

        # linear part of the design matrix
        d_2 = np.vstack([xval, yval, np.ones(len(xval))]).T

        # quadratic part of the scatter matrix
        s_1 = d_1.T @ d_1

        # combined part of the scatter matrix
        s_2 = d_1.T @ d_2

        # linear part of the scatter matrix
        s_3 = d_2.T @ d_2

        # reduced scatter matrix
        cnd = np.array(((0, 0, 2), (0, -1, 0), (2, 0, 0)), dtype=float)
        trc = -np.linalg.inv(s_3) @ s_2.T
        mat = np.linalg.inv(cnd) @ (s_1 + s_2 @ trc)

        # solve the eigen system
        eigvec = np.linalg.eig(mat)[1]

        # evaluate the coefficients
        con = 4 * eigvec[0] * eigvec[2] - eigvec[1] ** 2
        eiv_pos = eigvec[:, np.nonzero(con > 0)[0]]
        coefs = np.concatenate((eiv_pos, trc @ eiv_pos)).ravel()
        names = [f"beta{i}" for i in range(len(coefs))]
        self.betas = pd.DataFrame(coefs, index=names, columns=["CART. COEFS"])

        # get the axes angles
        # ref: http://www.geom.uiuc.edu/docs/reference/CRC-formulas/node28.html
        a, c, b = self.betas.values.flatten()[:3]

        # get the axes angles
        if c == 0:
            a0 = 0
        else:
            m0 = (b - a) / c
            m0 = (m0**2 + 1) ** 0.5 + m0
            m1 = -1 / m0
            a0 = np.arctan(m0)

        # We know that the two axes pass from the centre of the ellipsis
        # and we also know the angle of the major and minor axes.
        # Therefore the intercept of the fitting lines describing the two
        # axes can be found.
        x0, y0 = self.center
        i0 = y0 - x0 * m0
        i1 = y0 - x0 * m1

        # get two LinearRegression objects describing the two axes
        r0 = LinearRegression(y=np.array([y0, i0]), x=np.array([x0, 0]))
        r1 = LinearRegression(y=np.array([y0, i1]), x=np.array([x0, 0]))

        # get the crossings between the two axes and the ellipsis
        p0_0, p0_1 = self._get_crossings(r0)
        p1_0, p1_1 = self._get_crossings(r1)

        # generate the two axes
        ax0 = _Axis(x=[p0_0[0], p0_1[0]], y=[p0_0[1], p0_1[1]])
        ax1 = _Axis(x=[p1_0[0], p1_1[0]], y=[p1_0[1], p1_1[1]])

        # sort the axes
        if ax0.length < ax1.length:
            ax0, ax1 = ax1, ax0

        # store the axes
        self.axis_major = ax0
        self.axis_minor = ax1

    def _get_abc_by_x(self, x: float) -> tuple:
        """
        private method which calculates the values a, b and c used
        to extract the values of y given x.

        Parameters
        ----------
        x: float
            the given x value.

        Returns
        -------
        a, b, c: float
            the coefficients to be used for extracting the roots of a
            2nd order equation having y as unknown parameter.
        """
        x_ = float(x)
        a, b, c, d, e, f = self.betas.values.flatten()
        return c, b * x_ + e, f + a * x_**2 + d * x_

    def _get_abc_by_y(self, y: float) -> tuple:
        """
        private method which calculates the values a, b and c used
        to extract the values of x given y.

        Parameters
        ----------
        y: float
            the given y value.

        Returns
        -------
        a, b, c: float
            the coefficients to be used for extracting the roots of a
            2nd order equation having x as unknown parameter.
        """
        y_ = float(y)
        a, b, c, d, e, f = self.betas.values.flatten()
        return a, b * y_ + d, f + c * y_**2 + e * y_

    def _get_roots(self, a: float, b: float, c: float) -> tuple:
        """
        obtain the roots of a second order polynomial having form:
                a * x**2 + b * x + c = 0

        Parameters
        ----------
        a, b, c: float
            the coefficients of the polynomial.

        Returns
        -------
        x0, x1: float | None
            the roots of the polynomial. None is returned if the solution
            is impossible.
        """
        delta = b**2 - 4 * a * c
        if delta < 0:
            return None, None
        d = np.sqrt(delta)
        return (-b - d) / (2 * a), (-b + d) / (2 * a)

    def __call__(
        self,
        x: Union[np.ndarray, pd.DataFrame, list, int, float] = None,
        y: Union[np.ndarray, pd.DataFrame, list, int, float] = None,
    ) -> pd.DataFrame:
        """
        predict the x given y or predict y given x.

        Parameters
        ----------
        x OR y: (samples, 1) numpy array or pandas.DataFrame
            the array containing the dependent variable.

        Returns
        -------
        y OR x: (samples, 2) numpy array or pandas.DataFrame
            the array containing the dependent variable.

        Note
        ----
        only x or y can be provided. None is returned if the provided value
        lies outside the ellipsis boundaries.
        """
        # check the entries
        assert x is not None or y is not None, "'x' or 'y' must be provided."
        assert x is None or y is None, "only 'x' or 'y' must be provided."
        if x is not None:
            v = self._simplify(x, "X")
            fun = self._get_abc_by_x
            cols = ["Y0", "Y1"]
        else:
            v = self._simplify(y, "Y")
            fun = self._get_abc_by_y
            cols = ["X0", "X1"]
        assert v.shape[1] == 1, "Only 1D arrays can be provided."

        # calculate the values
        o = np.atleast_2d([self._get_roots(*fun(i)) for i in v.values])
        return pd.DataFrame(o, columns=cols, index=v.index).astype(float)

    @property
    def center(self) -> tuple:
        """
        get the center of the ellipsis as described here:
        https://mathworld.wolfram.com/Ellipse.html

        Returns
        -------
        x0, y0: float
            the coordinates of the centre of the ellipsis.
        """
        a, b, c, d, e = self.betas.values.flatten()[:-1]
        den = b**2 - 4 * a * c
        return (2 * c * d - b * e) / den, (2 * a * e - b * d) / den

    @property
    def centre(self) -> tuple:
        """
        get the center of the ellipsis as described here:
        https://mathworld.wolfram.com/Ellipse.html

        Returns
        -------
        x0, y0: float
            the coordinates of the centre of the ellipsis.
        """
        return self.center

    @property
    def area(self) -> float:
        """
        the area of the ellipsis
        """
        return np.pi * len(self.axis_major) * len(self.axis_minor)

    def _get_crossings(self, lr: LinearRegression) -> tuple:
        """
        get the crossings between the provided line and the ellipsis

        Parameters
        ----------
        m: float
            the slope of the axis line

        i: float
            the intercept of the axis line

        Returns
        -------
        p0, p1: tuple
            the coordinates of the crossing points. It returns None if
            the line does not touch the ellipsis.
        """
        i, m = lr.betas.values.flatten()
        a, b, c, d, e, f = self.betas.values.flatten()
        a_ = a + b * m + c * m**2
        b_ = b * i + 2 * m * i * c + d + e * m
        c_ = c * i**2 + e * i + f
        d_ = b_**2 - 4 * a_ * c_
        if d_ < 0:
            return None, None
        e_ = 2 * a_
        f_ = -b_ / e_
        g_ = (d_**0.5) / e_
        x0 = f_ - g_
        x1 = f_ + g_
        return (x0, x0 * m + i), (x1, x1 * m + i)

    @property
    def eccentricity(self) -> float:
        """
        return the eccentricity parameter of the ellipsis.
        """
        b = len(self.axis_minor) / 2
        a = len(self.axis_major) / 2
        return (1 - b**2 / a**2) ** 0.5

    @property
    def foci(self) -> tuple:
        """
        return the coordinates of the foci of the ellipses.

        Returns
        -------
        f0, f1: tuple
            the coordinates of the crossing points. It returns None if
            the line does not touch the ellipsis.
        """
        a = len(self.axis_major) / 2
        p = self.axis_major.angle
        x, y = a * self.eccentricity * np.array([np.cos(p), np.sin(p)])
        x0, y0 = self.centre
        return (x0 - x, y0 - y), (x0 + x, y0 + y)

    def is_inside(
        self,
        x: Union[int, float],
        y: Union[int, float],
    ) -> bool:
        """
        check whether the point (x, y) is inside the ellipsis.

        Parameters
        ----------
        x: float
            the x axis coordinate

        y: float
            the y axis coordinate

        Returns
        -------
        i: bool
            True if the provided point is contained by the ellipsis.
        """
        y0, y1 = self(x=x).values.flatten()
        return (y0 is not None) & (y > min(y0, y1)) & (y <= max(y0, y1))
