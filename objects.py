# MODELS MODULE


#! IMPORTS


import numpy as np
from pandas import MultiIndex, Index, DataFrame, Series


#! CLASSES


class TimeSeries(DataFrame):
    """
        Generate a n-dimensional TimeSeries object.

        Parameters
        ----------
        data: ndarray | list | DataFrame | Series | dict
            the data of the timeseries.
            If a DataFrame is provided, it is passed "as is" to the constructor.
            By default, a copy of it is passed, unless the "copy" argument is
            set to False.
            If ndarray, or list, they must be a 1D or 2D arrays with
            numeric-only data.

        time: list | ndarray | Index | None, optional
            If an Index is provided, it is passd as is to the _data object.
            If ndarray, or list, they must be a 1D or 2D arrays with
            numeric-only data.

            IMPORTANT! time is assumed to be provided in seconds.

            by default None

        dimensions: list | ndarray | Index | MultiIndex
            a MultiIndex already containing the following levels:
                - NAME
                - UNIT
                - AXIS
            alternatively, a 1D list or ndarray can be provided and used as
            axis level.

        unit: str | None, optional
            the unit of measurement of the dimensions. It must be the same
            for each axis.

            by default None

        name: str | None, optional
            the name of the timeseries.

            by default None

        attribute: str
            the object type

        copy: bool, optional
            if True, the data is copied before being used as constructor

            by default True
    """

    # ****** VARIABLES ****** #

    _mgr = DataFrame()._mgr  # required to avoid warnings or errors

    # ****** CONSTRUCTOR ****** #

    def __init__(
        self,
        data: np.ndarray | list | DataFrame | Series | dict,
        time: list | np.ndarray | Index | None = None,
        dimensions: list | np.ndarray | Index | MultiIndex | None = None,
        unit: str = "",
        name: str = "",
        attribute: str = "",
    ) -> None:
        if isinstance(data, (DataFrame, TimeSeries)):
            obj = self._read_frame(data)
            data, time, name, attribute, unit, dimensions = obj
        col = self._make_columns(dimensions, unit, name, attribute)
        idx = self._make_index(time)
        super().__init__(data=data, index=idx, columns=col)
        self._validate()

    # ****** METHODS ****** #

    def _read_frame(self, frame:DataFrame):
        cols = np.vstack(frame.columns.to_numpy()).T
        try:
            name, attribute, unit, dimensions = cols
            name = name[0]
            attribute = attribute[0]
            unit = unit[0]
            data = frame.values
            time = frame.index
        except Exception:
            raise(f"data is not castable to {self.__class__}.")
        return data, time, name, attribute, unit, dimensions

    def _make_columns(
        self,
        dimensions: list | np.ndarray | Index | MultiIndex,
        unit: str | None = None,
        name: str | None = None,
        attribute: str | None = None,
    ) -> MultiIndex:
        if dimensions is None:
            return None
        if isinstance(dimensions, MultiIndex):
            return dimensions
        if isinstance(dimensions, Index):
                ax = dimensions.to_numpy()
        elif isinstance(dimensions, list):
                ax = np.array(dimensions)
        elif isinstance(dimensions, np.ndarray):
                ax = dimensions
        else:
            cls = [MultiIndex, Index, list, np.ndarray]
            raise ValueError(f"{dimensions} must be an instance of {cls}.")
        if ax.ndim > 1:
            raise ValueError("dimensions must be a 1D iterable object.")
        if not isinstance(unit, str):
                raise ValueError("dimensions_unit must be a str object.")
        if not isinstance(name, str):
                raise ValueError("name must be a str object.")
        if not isinstance(attribute, str):
                raise ValueError("dtype must be a str object.")
        arr = [[name], [attribute], [unit], dimensions]
        return MultiIndex.from_product(arr)

    def _make_index(self, time: list | np.ndarray | Index) -> Index:
        if time is None:
            return None
        if isinstance(time, Index):
            return time
        if isinstance(time, list):
                ax = np.array(time)
        else:
                ax = time
        if not isinstance(ax, np.ndarray):
                raise ValueError("time must be a 1D iterable object.")
        if ax.ndim > 1:
                raise ValueError("time must be a 1D iterable object.")
        return Index(time)

    def _validate(self):
        def check_name(x:str):
            if not isinstance(x, str):
                raise ValueError(f"{x} must be a {str} instance.")
            if len(x.split(" ")) > 1:
                raise ValueError(f"{x} cannot have spaces.")

        self.index.set_names("TIME", inplace=True)
        if not self.index.dtype in (np.int_, np.float_):
            raise ValueError("time must be numeric-only.")
        cols = self.columns.to_frame()
        cols.applymap(check_name)
        if not isinstance(self.columns, MultiIndex) or cols.shape[1] != 4:
            raise TypeError(f"{cols} must be a 4 level MultiIndex")
        self.columns.set_names(
            ["NAME", "ATTRIBUTE", "UNIT", "AXIS"],
            inplace=True,
        )
        cols.columns = Index(self.columns.names)
        for value in self.dtypes.values:
            if not value in (np.int_, np.float_):
                raise ValueError(f"{value} must be numeric-only.")
        counts = np.unique(cols.values.astype(str), axis=0, return_counts=True)
        counts = counts[1]
        if any(i > 1 for i in counts):
            dup = cols.iloc[np.where(counts > 1)[0]].index.to_list()
            raise ValueError(f"The following columns have duplicates: {dup}")

    # ****** PROPERTIES ****** #

    @property
    def _constructor(self):
        return TimeSeries

    @property
    def _constructor_sliced(self):
        return Series

    @property
    def names(self):
        """return a dict separating the current object by name."""
        return {k: v for k, v in self.groupby(level=0, as_index=False)}

    @property
    def attributes(self):
        """return a dict separating the current object by attribute."""
        return {k: v for k, v in self.groupby(level=1, as_index=False)}

    @property
    def units(self):
        """
        return a dict separating the current object by unit of measurement.
        """
        return {k: v for k, v in self.groupby(level=2, as_index=False)}

    @property
    def dimensions(self):
        """return a dict separating the current object by axis name."""
        return {k: v for k, v in self.groupby(level=3, as_index=False)}


class BipolarEMG(TimeSeries):
    """
    Generate a n-dimensional TimeSeries object.

    Parameters
    ----------
    data: ndarray | list | DataFrame | Series | dict
        the data of the timeseries.
        If a DataFrame is provided, it is passed "as is" to the constructor.
        By default, a copy of it is passed, unless the "copy" argument is
        set to False.
        If ndarray, or list, they must be a 1D or 2D arrays with
        numeric-only data.

    time: list | ndarray | Index | None, optional
        If an Index is provided, it is passd as is to the _data object.
        If ndarray, or list, they must be a 1D or 2D arrays with
        numeric-only data.

        IMPORTANT! time is assumed to be provided in seconds.

        by default None

    unit: str | None, optional
        the unit of measurement of the dimensions. It must be the same
        for each axis.

        by default None

    name: str | None, optional
        the name of the timeseries.

        by default None
    """

    # ****** CONSTRUCTOR ****** #

    def __init__(
        self,
        data: np.ndarray | list | DataFrame | Series | dict,
        time: list | np.ndarray | Index | None = None,
        unit: str = "V",
        name: str = "",
    ) -> None:
        super().__init__(
            data=data,
            time=time,
            attribute="CHANNEL",
            dimensions=["AMPLITUDE"],
            unit=unit,
            name=name,
        )

    # ****** METHODS ****** #

    def _read_frame(self, frame:DataFrame):
        data, time, name, _, unit, _ = super()._read_frame(frame)
        return data, time, name, "CHANNEL", unit, ["AMPLITUDE"]
