


# IMPORTS



import numpy as np
import pandas as pd
import scipy.linalg as sl



# CLASSES



class LinearRegression():



    def __init__(self, y, x, order=1, fit_intercept=True):
        """
        Obtain the regression coefficients according to the Ordinary Least
        Squares approach.

        Input:
            y:              (2D column numpy array)
                            the array containing the dependent variable.

            x:              (2D array)
                            the array containing the indipendent variables.
                            The number of rows must equal the rows of y, while
                            each column will be a regressor (i.e. an indipendent
                            variable).

            order:          (int)
                            the order of the polynomial.

            fit_intercept:  (bool)
                            Should the intercept be included in the model?
                            Otherwise it will be set to zero.
        """

        # add the input parameters
        self.fit_intercept = fit_intercept

        # add the order of the fit
        self.order = order

        # correct the shape of y and x
        YY = self.__simplify__(y, 'Y', None)
        XX = self.__simplify__(x, 'X', self.order)
        txt = "'X' and 'Y' number of rows must be identical."
        assert XX.shape[0] == YY.shape[0], txt

        # add the ones for the intercept
        if self.fit_intercept:
            XX = np.hstack([np.ones((XX.shape[0], 1)), XX])

        # get the coefficients and intercept
        self._coefs = pd.DataFrame(
            data    = sl.pinv(XX.T.dot(XX)).dot(XX.T).dot(self.Y),
            index   = self.__IV_labels__,
            columns = self.__DV_labels__
            )



    @property
    def coefs(self):
        """
        vector of the regression coefficients.
        """
        return self._coefs



    def SSPE(self):
        """
        return the Sum of Square Product Error matrix
        """
        R = self.residuals()
        return R.T.dot(R)



    def cov_unscaled(self):
        """
        return the unscaled covariance (i.e. without multiplication for the
        variance term) of the coefficients.
        """
        if self.fit_intercept:
            I = pd.DataFrame(
                data  = {'Intercept': np.tile(1, self.X.shape[0])},
                index = self.X.index
                )
            X = pd.concat([I, self.X], axis=1)
        else:
            X = self.X
        return pd.DataFrame(
            data    = sl.inv(X.T.dot(X)),
            index   = X.columns,
            columns = X.columns
            )



    def residuals(self):
        """
        obtain the residuals of the current regression model.
        """
        return self.Y - self.predict(self.X)



    @property
    def __IV_labels__(self):
        """
        return the labels for the regressors.
        """
        out = []
        if self.fit_intercept:
            out += ['Intercept']
        if isinstance(self.X, pd.DataFrame):
            X = self.X.columns.tolist()
        else:
            N = np.arange(self.X.shape[1])
            X = ['X{}'.format(i) for i in N]
        out += X
        for i in np.arange(2, self.order + 1):
            for c in X:
                out += [c + "^{}".format(i)]
        return out



    @property
    def __DV_labels__(self):
        """
        return the labels for the dependent variables.
        """
        if isinstance(self.Y, pd.DataFrame):
            return self.Y.columns.to_numpy().tolist()
        return ['Y{}'.format(i) for i in np.arange(self.Y.shape[1])]



    def __simplify__(self, v, name, order):
        """
        internal method to check entries in the constructor.
        """
        txt = "'{}' must be a pandas.DataFrame, a numpy.ndarray or None."
        txt = txt.format(name)
        if isinstance(v, pd.DataFrame):
            XX = v
        else:
            if isinstance(v, np.ndarray):
                XX = pd.DataFrame(np.squeeze(v))
            elif v is None:

                # try to get the shape from the other parameter
                try:
                    N = self.Y.shape[0]
                except Exception:
                    try:
                        N = self.X.shape[0]
                    except Exception:
                        raise ValueError(txt)
                XX = np.atleast_2d([[] for i in np.arange(N)])

                # try to convert to the most similar pandas.DataFrame
                # to the other parameter.
                try:
                    IX = self.Y.index
                except Exception:
                    try:
                        IX = self.X.index
                    except Exception:
                        IX = pd.Index(np.arange(N))
                N = np.arange(XX.shape[1]) + 1
                CX = pd.Index(name + "{}".format(i) for i in N)
                XX = pd.DataFrame(XX, index=IX, columns=CX)
            else:
                raise NotImplementedError(txt)
        setattr(self, name, XX)
        ZZ = XX.values
        if order is not None:
            for i in np.arange(2, self.order + 1):
                for c in np.arange(XX.shape[1]):
                    K = np.atleast_2d(ZZ[:, c] ** i)
                    ZZ = np.hstack([ZZ, K.T])
        return ZZ



    def copy(self):
        """
        copy the current instance
        """
        return LinearRegression(self.Y, self.X, self.fit_intercept)



    def predict(self, x):
        """
        predict the fitted Y value according to the provided x.
        """
        X = self.__simplify__(x, 'x', self.order)
        n = self.coefs.shape[0] - 1
        assert X.shape[1] == n, "'X' must have {} columns.".format(n)
        Z = X.dot(self.coefs.values[1:]) + self.coefs.values[0]
        if isinstance(x, pd.DataFrame):
            idx = x.index
        else:
            idx = np.arange(X.shape[0])
        return pd.DataFrame(Z, index=idx, columns=self.__DV_labels__)



    def DF(self):
        """
        return the degrees of freedom of the model.
        """
        return self.Y.shape[0] - self.coefs.shape[0]



    def SS(self):
        """
        calculate the sum of squares of the fitted model.
        """
        P = self.predict(self.X).values.flatten()
        return np.sum((P - np.mean(P, 0)) ** 2, 0)



    def R2(self):
        """
        calculate the R-squared of the fitted model.
        """
        N = np.var(self.residuals().values.flatten())
        D = np.var(self.Y.values.flatten())
        return 1 - N / D



    def R2_adjusted(self):
        """
        calculate the Adjusted R-squared of the fitted model.
        """
        den = (len(self.Y) - self.DF() - 1)
        return 1 - (1 - self.R2) * (self.Y.shape[0] - 1) / den


    def RMSE(self):
        """
        Get the Root Mean Squared Error
        """
        return np.sqrt(np.mean(self.residuals().values.flatten() ** 2))



    def toString(self, digits=3):
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
        txt = ""
        ivs = self.coefs.columns.to_numpy()
        dvs = self.coefs.index.to_numpy()
        frm = "{:+." + str(digits) + "f}"
        for iv in ivs:
            line = "{} = ".format(iv)
            for dv in dvs:
                v = self.coefs.loc[dv, iv]
                if dv == "Intercept":
                    line += frm.format(v)
                else:
                    s = " " + frm + " {}"
                    line += s.format(v, dv)
            txt += line + "\n"
        return txt



class PowerRegression(LinearRegression):



    def __init__(self, y, x):
        """
        Obtain the regression coefficients according to the Ordinary Least
        Squares approach.

        Input:
            y:              (2D column numpy array)
                            the array containing the dependent variable.

            x:              (2D array)
                            the array containing the indipendent variables.
                            The number of rows must equal the rows of y, while
                            each column will be a regressor (i.e. an indipendent
                            variable).
        """

        # correct the shape of y and x
        YY = np.log(self.__simplify__(y, 'Y', None))
        XX = np.log(self.__simplify__(x, 'X', None))
        txt = "'X' and 'Y' number of rows must be identical."
        assert XX.shape[0] == YY.shape[0], txt
        assert XX.shape[1] == 1, "'X' must have just 1 column."

        # add the ones for the intercept
        XX = np.hstack([np.ones((XX.shape[0], 1)), XX])

        # get the coefficients
        coefs = sl.pinv(XX.T.dot(XX)).dot(XX.T).dot(YY)
        coefs[0] = np.e ** coefs[0]

        # get the coefficients and intercept
        self._coefs = pd.DataFrame(
            data    = coefs,
            index   = self.__IV_labels__,
            columns = self.__DV_labels__
            )



    def copy(self):
        """
        copy the current instance
        """
        return PowerRegression(self.Y, self.X)



    def predict(self, x):
        """
        predict the fitted Y value according to the provided x.
        """
        X = self.__simplify__(x, 'x', None)
        assert X.shape[1] == 1, "'X' must have 1 column."
        Z = self.coefs.loc['a'].values * X ** self.coefs.loc['b'].values
        if isinstance(x, pd.DataFrame):
            idx = x.index
        else:
            idx = np.arange(X.shape[0])
        return pd.DataFrame(Z, index=idx, columns=self.__DV_labels__)



    @property
    def __IV_labels__(self):
        """
        return the labels for the regressors.
        """
        return ['a', 'b']



    def toString(self, digits=3):
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
        txt = "{} = " + frm + " {} ^ " + frm
        return txt.format(
            self.coefs.columns.to_numpy()[0],
            self.coefs.loc['a'].values.flatten()[0],
            self.X.columns.to_numpy()[0],
            self.coefs.loc['b'].values.flatten()[0]
            )
