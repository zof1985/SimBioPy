# MODELS MODULE


#! IMPORTS


import numpy as np
from pandas import MultiIndex, Index, DataFrame, Series


#! CLASSES


class TimeSeries():
    """
        Generate a n-dimensional TimeSeries object.

        Parameters
        ----------
        data: np.ndarray | list | DataFrame | Series | dict
            the data of the timeseries.
            If a DataFrame is provided, it is passed "as is" to the constructor.
            By default, a copy of it is passed, unless the "copy" argument is
            set to False.
            If np.ndarray, or list, they must be a 1D or 2D arrays with
            numeric-only data.

        time: list | np.ndarray | Index | None, optional
            If an Index is provided, it is passd as is to the _data object.
            If np.ndarray, or list, they must be a 1D or 2D arrays with
            numeric-only data.

            IMPORTANT! time is assumed to be provided in seconds.

            by default None

        axes: list | np.ndarray | Index | MultiIndex
            a MultiIndex already containing the following levels:
                - NAME
                - UNIT
                - AXIS
            alternatively, a 1D list or ndarray can be provided and used as
            axis level.

        unit: str | None, optional
            the unit of measurement of the axes. It must be the same for each
            axis.

            by default None

        name: str | None, optional
            the name of the timeseries.

            by default None

        copy: bool, optional
            if True, the data is copied before being used as constructor

            by default True
    """

    # ****** VARIABLES ****** #

    _data = DataFrame()

    # ****** CONSTRUCTOR ****** #

    def __init__(
        self,
        data: np.ndarray | list | DataFrame | Series | dict,
        time: list | np.ndarray | Index | None = None,
        axes: list | np.ndarray | Index | MultiIndex | None = None,
        unit: str = "",
        name: str = "",
        copy: bool = True,
    ) -> None:
        if isinstance(data, DataFrame):
            self._data = DataFrame(data, copy=copy)
        else:
            col = self._make_columns(axes, unit, name)
            idx = self._make_index(time)
            self._data = DataFrame(data=data, index=idx, columns=col)
        self._validate()

    # ****** METHODS ****** #

    def __repr__(self):
        return self._data.__repr__()

    def __str__(self):
        return self._data.__str__()

    def __getattr__(self, key):
        obj = self._data if key == "_data" else self._data.__getattr__(key)
        return obj

    def __setattr__(self, key, value):
        if key == "_data":
            super().__setattr__(key, value)
        else:
            self._data.__setattr__(key, value)

    def __getitem__(self, obj):
        return self._data.__getitem__(obj)

    def __setitem__(self, obj, value):
        self._data.__setitem__(obj, value)

    def _make_columns(
        self,
        axes: list | np.ndarray | Index | MultiIndex,
        unit: str | None = None,
        name: str | None = None,
    ) -> MultiIndex:
        """
        private method used to generate the _data columns

        Parameters
        ----------
        axes : list | np.ndarray | Index | MultiIndex
            a MultiIndex already containing the following levels:
                - NAME
                - UNIT
                - AXIS
            alternatively, a 1D list or ndarray can be provided and used as
            axis level.

        unit : str | None, optional
            the unit of measurement of the axes. It must be the same for each
            axis.

            by default None

        name : str | None, optional
            the name of the timeseries.

            by default None

        Returns
        -------
        MultiIndex
            the columns to be provided to the _data variable.

        Raises
        ------
        ValueError
            if axes is not a MultiIndex and it has more than 1 dimension.
        """
        if isinstance(axes, MultiIndex):
            return axes
        else:
            if isinstance(axes, Index):
                ax = axes.to_numpy()
            elif isinstance(axes, list):
                ax = np.array(axes)
            if ax.ndim > 1:
                raise ValueError("axes must be a 1D iterable object.")
            if not isinstance(unit, str):
                raise ValueError("axes_unit must be a str object.")
            if not isinstance(name, str):
                raise ValueError("name must be a str object.")
            return MultiIndex.from_product([[name], [unit], axes])

    def _make_index(self, time: list | np.ndarray | Index) -> Index:
        """
        create the index for the _data object.

        Parameters
        ----------
        time: list | np.ndarray | Index | None, optional
            If an Index is provided, it is passd as is to the _data object.
            If np.ndarray, or list, they must be a 1D or 2D arrays with
            numeric-only data.

            by default None

        Returns
        -------
        Index
            the time index to be provided to the _data variable.

        Raises
        ------
        ValueError
            if time is not an Index and it has more than 1 dimension.
        """
        if isinstance(time, Index):
            return time
        else:
            if isinstance(time, list):
                ax = np.array(time)
            else:
                ax = time
            if not isinstance(time, np.ndarray):
                raise ValueError("time must be a 1D iterable object.")
            if ax.ndim > 1:
                raise ValueError("time must be a 1D iterable object.")
            return Index(time)


    def _validate(self):
        """validate the _data object"""
        self._data.index._name = "TIME"
        self._data.columns._name = ["NAME", "UNIT", "AXIS"]
        for value in self._data.dtypes.values:
            if not value in (np.int_, np.float_):
                raise ValueError(f"{value} must be numeric-only.")
        if not self._data.index.dtype in (np.int_, np.float_):
            raise ValueError(f"time must be numeric-only.")
