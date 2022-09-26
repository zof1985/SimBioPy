"""
A suite of classes and functions to handle 3D models combining kinematic,
kinetic and EMG data.
"""


#! IMPORTS

import sys
import warnings

from typing import Any
from typing_extensions import Self
from numpy.typing import NDArray
from PySide2 import QtWidgets as widgets
from PySide2 import QtCore as qtcore

import numpy as np
import pandas as pd
import plotly.express as px

from .qtwidgets_work_in_progress import Model3DWidget


#! CLASSES


class TimeSeries:
    """basic object allowing to deal with complex 3D and EMG models"""

    # ****** VARIABLES ****** #

    _category = None
    _dimensions = None
    _properties = None
    _data = pd.DataFrame()
    _n = 0

    # ****** CONSTRUCTOR ****** #

    def __init__(self, data: pd.DataFrame) -> None:
        """constructor"""
        self._validate(data)

    # ****** OVERRIDDEN METHODS ****** #

    def __iter__(self) -> pd.DataFrame:
        """class iterator"""
        self._n = 0
        return self

    def __next__(self) -> pd.DataFrame | StopIteration:
        """next handler"""
        if self._n < len(self.names):
            self._n += 1
            return self._data.loc[:, self.names[self._n - 1]]
        raise StopIteration

    def __getattr__(self, attr: str) -> pd.DataFrame:
        """get the required attribute"""
        self._check_value(attr, str)
        if attr in self.names + self.properties + self.dimensions + self.units:
            return self._data.loc[:, attr]
        else:
            return getattr(self, attr)

    def __setattr__(self, attr: str, value: Any) -> pd.DataFrame:
        """set the required attribute"""
        self._check_value(attr, str)
        if attr in self.names + self.properties + self.dimensions + self.units:
            self._data.loc[:, attr] = value
        else:
            return setattr(self, attr, value)

    # ****** PROPERTIES ****** #

    @property
    def names(self) -> list:
        """return the items contained by the dataframe."""
        return np.unique(self._data.columns.to_numpy()[0].flatten()).tolist()

    @property
    def categories(self) -> list:
        """return the categories contained by the dataframe."""
        return np.unique(self._data.columns.to_numpy()[1].flatten()).tolist()

    @property
    def properties(self) -> list:
        """return the properties contained by the dataframe."""
        return np.unique(self._data.columns.to_numpy()[2].flatten()).tolist()

    @property
    def units(self) -> list:
        """return the units contained by the dataframe."""
        return np.unique(self._data.columns.to_numpy()[3].flatten()).tolist()

    @property
    def dimensions(self) -> list:
        """return the dimensions contained by the dataframe."""
        return np.unique(self._data.columns.to_numpy()[4].flatten()).tolist()

    @property
    def sampling_frequency(self) -> float:
        """return the sampling frequency contained by the dataframe."""
        return np.mean(np.diff(self._data.index.to_numpy()))

    @property
    def psd(self) -> pd.DataFrame:
        """return the power spectral density of the available signal(s)."""
        vals = np.atleast_2d(self._data.values)
        pows = abs(np.fft.rfft(vals - np.mean(vals, 0)) / self._data.shape[0])
        pows = np.concatenate([pows[0], 2 * pows[1:-1], pows[-1]]) ** 2
        frqs = np.linspace(0, self.sampling_frequency // 2, self._data.shape[0])
        return pd.DataFrame(
            data=pows,
            columns=self._data.columns,
            index=pd.Index(frqs, name="Frequency"),
        )

    @property
    def ndim(self) -> int:
        """return the number of dimensions."""
        return len(self._dimensions)

    @property
    def reference_columns(self) -> pd.DataFrame:
        """
        private method used to define the reference dataframe of this obejct.
        """
        out = []
        for prop in self.properties:
            obj = [
                np.tile(self._category, self.ndim),
                np.tile(prop, self.ndim),
                self._dimensions,
            ]
            obj = pd.DataFrame(
                data=pd.vstack(pd.atleast_2d(obj)),
                index=["CATEGORY", "PROPERTY", "DIMENSION"],
            )
            out += [obj]
        return pd.concat(out, axis=1, ignore_index=True)

    # ****** METHODS ****** #

    def isolate(self, copy=True) -> pd.DataFrame:
        """return the samples isolated by the actual accessor."""
        self._check_value(copy, bool)
        return self._data.copy() if copy else self._data

    def simplify(self, copy=True) -> Self:
        """return a simplified version of the model without unused indices."""
        out = self.isolate(copy)
        out.columns.remove_unused_levels()
        return out.drop_duplicates()

    @staticmethod
    def to_wide(dataframe: pd.DataFrame) -> pd.DataFrame:
        """
        unpack a long-format dataframe to the standard wide format.

        Parameters
        ----------
        dataframe: pandas.DataFrame
            the dataframe in long format. It must have the following columns:
        """
        keys = ["NAME", "TYPE", "PROPERTY", "UNIT", "DIMENSION"]
        return dataframe.pivot_table("VALUE", index="TIME", columns=keys)

    def to_long(self) -> pd.DataFrame:
        """return the current model as long-format dataframe"""
        dataframe = self._data.copy()
        dataframe.insert(0, "TIME", dataframe.index.to_numpy())
        return dataframe.melt(id_vars="TIME", value_name="VALUE")

    def plot_2d(
        self,
        mode: str = "lines",
        renderer: str | None = None,
        **opts,
    ) -> None:
        """
        return a plotly Figure object representing the model.

        Parameters
        ----------
        mode: str
            determine the style of the plot. "lines" defines a line plot,
            "scatter" denotes a scatter plot.

        renderer: str | None
            the renderer to be used for showing the figure i.e. None (browser)
            or "notebook" for notebooks.

        **opts: object
            list of keyworded options passed to the plotly.express function
            used to generate the figure.
        """

        # get the data
        pdata = self.to_long()
        cols = [f"{x} ({y})" for x, y in zip(pdata.DIMENSION, pdata.UNIT)]
        pdata.loc[pdata.index, "DIMENSION"] = cols
        pdata = pdata.drop("UNIT", axis=1)

        # plot the figure
        self._check_value(mode, str)
        if mode == "lines":
            fun = px.line
        elif mode == "scatter":
            fun = px.scatter
        else:
            raise ValueError("mode must be a 'lines' or 'scatter'.")
        fig = fun(
            data_frame=pdata,
            x="TIME (s)",
            y="VALUE",
            color="DIMENSION",
            facet_col="PROPERTY",
            facet_row="NAME",
            **opts,
        )

        # update the layout
        fig.for_each_yaxis(lambda a: a.update(text=""))
        fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))

        # show
        fig.show(renderer=renderer)

    def _check_value(self, obj: Any, cls: Any) -> None:
        """
        check whether obj is instance of the provided class.

        Parameters
        ----------
        obj: Object
            any object.

        cls: Object
            the class instance
        """
        if not isinstance(obj, cls):
            raise TypeError(f"'{obj} must be an instance of {cls}.")

    def _validate(self, obj: object) -> None:
        """
        check whether the input object can be integrated within the model.

        Parameters
        ----------
        obj: object
            Any python object. In order to be a valid element for the model,
            the object must be a pandas dataframe having:
            - numerical-only values
            - simple numerical index
            - MultiIndex columns with structure:
                NAME, CATEGORY, PROPERTY, UNIT, DIMENSION
        """
        # set the numpy accepted dtypes
        dtypes = set("if")

        # check the object validity
        valid_cols = []
        if not isinstance(obj, pd.DataFrame):
            valid_idxs = []
        else:
            index = obj.index.to_numpy()
            values = obj.values
            columns = obj.columns.to_frame()
            if not (
                index.dtype.kind in dtypes
                and values.dtype.kind in dtypes
                and columns.shape[0] == 5
                and np.all(~columns.isna().values)
            ):
                valid_idxs = []
            else:

                # get the valid columns
                labels = ["NAME", "CATEGORY", "PROPERTY", "UNIT", "DIMENSION"]
                columns.index = pd.Index(labels)
                columns.drop("UNIT", axis=0, inplace=True)
                for group in columns.T.groupby(level=0):
                    if np.all(group[1].isin(self.reference_columns).values):
                        valid_cols += group[1].index.to_list()
                valid_cols = obj.columns[valid_cols]

                # get the valid indices
                valid_idxs = obj[valid_cols].dropna().drop_duplicates().index

        # set data
        self._data = obj.loc[valid_idxs, valid_cols]


@pd.api.extensions.register_dataframe_accessor("emg")
class EmgChannel(TimeSeries):
    """handle EMG channels stored within the dataframe"""

    # ****** VARIABLES ****** #

    _category = "emg"
    _dimensions = ["amplitude"]
    _properties = ["channel"]

    # ****** CONSTRUCTOR ****** #

    def __init__(self, data: pd.DataFrame) -> None:
        """constructor"""
        super().__init__(data)

    # ****** PROPERTIES ****** #

    def mean_frequency(self) -> pd.Series:
        """return the mean power frequency of the signal in Hz"""
        psd = self.psd
        frq = np.atleast_2d(psd.index.to_numpy()).T
        return (psd * np.ones(psd.shape) * frq).sum(0) / psd.sum(0)

    def median_frequency(self) -> pd.Series:
        """return the median power frequency of the signal in Hz"""
        psd = self.psd
        frq = psd.index.to_numpy()
        return psd.apply(
            lambda x: frq[np.where(np.cumsum(x) / np.sum(x) >= 0.5)[0]],
        )

    # ****** METHODS ****** #

    def center(self, inplace: bool = False) -> Any | None:
        """
        center the signal around the mean value.

        Parameters
        ----------
        inplace: bool
            if True the signal is centered inplace. Otherwise a copy of the
            signal is returned.

        Returns
        -------
        channel: pd.DataFrame
            if inplace is False, this will be the centered signal.
        """
        self._check_value(inplace, bool)
        if inplace:
            self._data -= self._data.mean(0)
        else:
            return self._data - self._data.mean(0)


class TimeSeries3D(TimeSeries):
    """handle EMG channels stored within the dataframe"""

    # ****** CONSTRUCTOR ****** #

    def __init__(self, data: pd.DataFrame) -> None:
        """constructor"""
        super().__init__(data)

    # ****** METHODS ****** #

    def plot_3d(self, vertical_axis: str = "Y") -> None:
        """
        return a 3D representation of the model.

        Parameters
        ----------
        vertical_axis: str
            the dimension denoting the vertical axis.
        """

        # check the input
        self._check_value(vertical_axis, str)
        if not vertical_axis in self.dimensions:
            raise ValueError(f"{vertical_axis} not found in {self.dimensions}")

        # highdpi scaling
        high_dpi = qtcore.Qt.AA_EnableHighDpiScaling
        widgets.QApplication.setAttribute(high_dpi, True)
        high_pixmaps = qtcore.Qt.AA_UseHighDpiPixmaps
        widgets.QApplication.setAttribute(high_pixmaps, True)

        # app generation
        app = widgets.QApplication(sys.argv)
        widget = Model3DWidget(self, vertical_axis=vertical_axis)
        widget.show()
        app.exec_()

    def move_to(
        self,
        reference_frame: pd.DataFrame,
        inplace: bool = False,
    ) -> pd.DataFrame | None:
        """
        Rotate the current point(s) by the provided reference frame(s).

        Parameters
        ----------
        reference_frame: pd.DataFrame
            one or more pandas DataFrame being valid ReferenceFrame objects.

        inplace: bool
            if True the signal is centered inplace. Otherwise a copy of the
            signal is returned.

        Returns
        -------
        points: pd.DataFrame
            if inplace is False, this will be the points aligned to the
            provided reference frame(s).

        Note
        ----
        In case more Reference Frames are included in the reference_frame
        dataframe, all of them are concatenated.
        """

        # check the entries
        self._check_value(reference_frame, pd.DataFrame)
        if len(reference_frame.reference_frame.names) == 0:
            msg = "No reference frame has been found in the provided dataframe."
            warnings.warn(msg)
        self._check_value(inplace, bool)

        # concatenate the rotations
        new = self._data if inplace else self._data.copy()
        for frame in reference_frame.reference_frame:
            origin = frame.reference_frame.origin.values
            rmat = frame.reference_frame.rotation_matrix
            for name in self.names:
                data = (new.loc[:, name] - origin).values.T
                new.loc[:, name] = rmat.dot(data)

        # return
        if not inplace:
            return new


@pd.api.extensions.register_dataframe_accessor("point3d")
class Point3D(TimeSeries3D):
    """handle EMG channels stored within the dataframe"""

    # ****** VARIABLES ****** #

    _category = "point3d"
    _dimensions = ["X", "Y", "Z"]
    _properties = ["coordinates"]

    # ****** CONSTRUCTOR ****** #

    def __init__(self, data: pd.DataFrame) -> None:
        """constructor"""
        super().__init__(data)


@pd.api.extensions.register_dataframe_accessor("reference_frame")
class ReferenceFrame(TimeSeries3D):
    """handle dataframes as 3D space reference frames."""

    # ****** VARIABLES ****** #

    _category = "reference_frame"
    _dimensions = ["X", "Y", "Z"]
    _properties = ["origin", "p0", "p1", "p2"]

    # ****** CONSTRUCTOR ****** #

    def __init__(self, data: pd.DataFrame) -> None:
        """constructor"""
        super().__init__(data)

    # ****** PROPERTIES ****** #

    @property
    def rotation_matrix(self) -> NDArray:
        """
        return a 3D array representing the rotation_matrix moving from --> to
        the object for each sample.
        """
        if len(self.names) > 1:
            msg = "'rotation_matrix' can be provided only for single "
            msg += "ReferenceFrame objects."
            raise TypeError(msg)
        pnt0 = np.expand_dims(self.p0.reference_frame.simplify().values, 1)
        pnt1 = np.expand_dims(self.p1.reference_frame.simplify().values, 1)
        pnt2 = np.expand_dims(self.p2.reference_frame.simplify().values, 1)
        pnts = np.concat([pnt0, pnt1, pnt2], axis=1)
        mods = np.sum(pnts**2, axis=2) * np.ones_like(pnts)
        return np.atleast_3d(map(np.linalg.pinv, pnts / mods))


@pd.api.extensions.register_dataframe_accessor("force_platform")
class ForcePlatform(TimeSeries3D):
    """handle dataframes as 3D force platforms."""

    # ****** VARIABLES ****** #

    _category = "force_platform"
    _dimensions = ["X", "Y", "Z"]
    _properties = ["p0", "p1", "p2", "p3", "cop", "force", "moment"]

    # ****** CONSTRUCTOR ****** #

    def __init__(self, data: pd.DataFrame) -> None:
        """constructor"""
        super().__init__(data)

    # ****** PROPERTIES ****** #

    @property
    def module(self) -> pd.DataFrame:
        """
        return a pandas DataFrame containing the module of the force vector(s)
        """
        out = []
        for name in self.names:
            frz = self._data.loc[:, name].force_platform.force
            unit = np.unique(frz.columns.to_numpy()[-2, :])[0]
            cols = (name, self._category, "module", unit, "|XYZ|")
            cols = pd.MultiIndex.from_tuples([cols])
            out += [pd.DataFrame(np.sum(frz.values**2, 1), columns=cols)]
        return pd.concat(out, axis=1)

    # ****** METHODS ****** #

    def join(self, vertical_axis:str="Y") -> pd.DataFrame:
        """
        join together multiple force platforms into one.

        Parameters
        ----------
        vertical_axis: str
            the dimension denoting the vertical axis.
        """

        # check the input
        self._check_value(vertical_axis, str)
        if not vertical_axis in self.dimensions:
            raise ValueError(f"{vertical_axis} not found in {self.dimensions}")

        # merge the points
        points = []
        for name in self.names:
            for point in ['p0', 'p1', 'p2', 'p3']:
                points += [self._data.loc[:, name].loc[:, point]]

        # get the dimensions excluding the vertical axis
        dims = [i for i in self.dimensions if i != vertical_axis]

        # for each sample get the extreme points
        new_pnts = []
        pnt_cols = self._data.loc[:, ['p0', 'p1', 'p2', 'p3']].columns
        unit = self._data.loc[:, 'p0']
        for i in self._data.index.to_numpy():
            new_samples = []
            for dim in dims:
                data = self._data.loc[i, pnt_cols]
                subs = data.loc[:, [dim]]
                vals = subs.values.flatten()
                minv = np.argmin(vals)
                maxv = np.argmax(vals)

                # get the points
                p0_name = subs.columns[minv].to_numpy()[2]
                p0_crds = data.loc[:, p0_name].copy()
                p1_name = subs.columns[maxv].to_numpy()[2]
                p1_crds = data.loc[:, p1_name].copy()
                new_samples += [p0_crds, p1_crds]

            # update the columns name
            for j, sample in enumerate(new_samples):
                sample.columns = pd.MultiIndex.from_tuples([("resultant", "ForcePlatform3D", f"p{j}", )])

        name = "resultant"
        cat = "force_platform"
        dims = [i for i in self.dimensions if i != vertical_axis]
        pnt_0 = points.loc[:, [dims[0]]]
        pnt_1 = points.loc[:, [dims[1]]]
        p0 = self._data.loc[:, "p0"]
        p1 = self._data.loc[:, "p1"]
        p2 = self._data.loc[:, "p2"]
        p3 = self._data.loc[:, "p3"]

        for i in points.index:
            val_0 = pnt_0.loc[i].values.flatten()
            p0_col = pnt_0.columns[np.argmin(val_0)]
            p1_col = pnt_0.columns[np.argmax(val_0)]
            val_1 = pnt_1.loc[i].values.flatten()
            p2_col = pnt_0.columns[np.argmin(val_1)]
            p3_col = pnt_0.columns[np.argmax(val_1)]

        forces = self._data.loc[:, ['force']]
        moments = self._data.loc[:, ['moment']]
        p0s = pd.DataFrame()
        p1s = pd.DataFrame()
        p2s = pd.DataFrame()
        p3s = pd.DataFrame()
        res = pd.DataFrame()
        trq = pd.DataFrame()
        p0_i = np.argmin(points.loc[:, dims[0]].values, axis=1)
        p1_i = np.argmax(points.loc[:, dims[0]].values, axis=1)
        p2_i = np.argmin(points.loc[:, dims[1]].values, axis=1)
        p3_i = np.argmax(points.loc[:, dims[1]].values, axis=1)
        p0 =
        unit = plt.columns[0].to_numpy()[-2, 0]
                col = pd.MultiIndex.from_tuples([(name, cat, prop, unit, dim)])
                if prop in ["force", "moment"]:
                    val = np.atleast_2d(np.sum(plt.values, 1)).T
                elif prop in ["p0", "p1", "p2", "p3"]:

                    val = np.atleast_2d(np.sum(plt.values, 1)).T

            # get the joined
            pnt = pd.concat([plt.p0, plt.p1, plt.p2, plt.p3], axis=1)
            for grp in pnt.groupby(level=4, as_index=False):
                p0 = grp[1].columns[np.argmin(grp[1].values, axis=1)]
                p1 = grp[1].columns[np.argmax(grp[1].values, axis=1)]

            if pnts.shape[0] == 0:
            pnts = pnt
            forces += [plt.force]
            moments += [plt.moments]


class Model3D:
    """
    accessor allowing to deal with complex 3D and EMG models.
    """

    # ****** VARIABLES ****** #

    _data = pd.DataFrame()

    # ****** CONSTRUCTOR ****** #

    def __init__(self, data: pd.DataFrame) -> None:
        """constructor"""
        self._data = pd.DataFrame()
        self.add(data)

    # ****** PROPERTIES ****** #

    @property
    def points(self) -> Self:
        """return a view of the point data contained within the model."""
        col = self._data.columns
        col = col[np.where(col.to_numpy()[1].flatten() == "Point3D")[0]]
        idx = self._data[col].drop_duplicates().index
        return self._data.loc[idx, col]

    @property
    def force_platforms(self) -> Self:
        """return a Model3D subset containing only ForcePlatform3D elements."""
        col = self._data.columns
        col = col[np.where(col.to_numpy()[1].flatten() == "ForcePlatform3D")[0]]
        idx = self._data[col].drop_duplicates().index
        return self._data.loc[idx, col]

    @property
    def segments(self) -> Self:
        """return a Model3D subset containing only Segment3D elements."""
        col = self._data.columns
        col = col[np.where(col.to_numpy()[1].flatten() == "Segment3D")[0]]
        idx = self._data[col].drop_duplicates().index
        return self._data.loc[idx, col]

    @property
    def emgs(self) -> Self:
        """return a Model3D subset containing only EMG elements."""
        col = self._data.columns
        col = col[np.where(col.to_numpy()[1].flatten() == "EMGChannel")[0]]
        idx = self._data[col].drop_duplicates().index
        return self._data.loc[idx, col]

    @property
    def names(self) -> NDArray:
        """return the items contained by the model."""
        return np.unique(self._data.columns.to_numpy()[0].flatten())

    @property
    def types(self) -> NDArray:
        """return the types contained by the model."""
        return np.unique(self._data.columns.to_numpy()[1].flatten())

    @property
    def properties(self) -> NDArray:
        """return the properties contained by the model."""
        return np.unique(self._data.columns.to_numpy()[2].flatten())

    @property
    def units(self) -> NDArray:
        """return the units contained by the model."""
        return np.unique(self._data.columns.to_numpy()[3].flatten())

    @property
    def dimensions(self) -> NDArray:
        """return the dimensions contained by the model."""
        return np.unique(self._data.columns.to_numpy()[4].flatten())

    @property
    def sampling_frequency(self) -> float:
        """return the sampling frequency contained by the model."""
        return np.mean(np.diff(self._data.index.to_numpy()))

    # ****** METHODS ****** #

    def simplify(self) -> Self:
        """return a simplified version of the model without unused indices."""
        out = self._data.dropna()
        out.columns.remove_unused_levels()
        return out.drop_duplicates()

    def add(self, *args: pd.DataFrame | Self) -> None:
        """
        add objects to the model.

        Parameters
        ----------
        args: pd.DataFrame | Model3D
            any Model3D object or pandas dataframe having:
            - numerical-only values
            - simple numerical index
            - MultiIndex columns with structure: OBJECT_NAME, OBJECT_TYPE,
            PROPERTY_NAME, PROPERTY_UNIT, DIMENSION_NAME

        NOTE
        ----
        if a pandas dataframe is provided, the index is used as time reference
        and is assumed to be given in seconds.
        """
        names = ["NAME", "TYPE", "PROPERTY", "UNIT", "DIMENSION"]
        for arg in args:
            self._validate(arg)
            arg.columns.set_names(names, inplace=True)
            arg.index.set_names("TIME", inplace=True)
            self._data = pd.concat([self._data, arg], axis=1)

    @staticmethod
    def to_wide(dataframe: pd.DataFrame) -> pd.DataFrame:
        """
        unpack a long-format dataframe to the standard wide format.

        Parameters
        ----------
        dataframe: pandas.DataFrame
            the dataframe in long format. It must have the following columns:
        """
        keys = ["NAME", "TYPE", "PROPERTY", "UNIT", "DIMENSION"]
        return dataframe.pivot_table("VALUE", index="TIME", columns=keys)

    def to_long(self) -> pd.DataFrame:
        """return the current model as long-format dataframe"""
        dataframe = self._data.copy()
        dataframe.insert(0, "TIME", dataframe.index.to_numpy())
        return dataframe.melt(id_vars="TIME", value_name="VALUE")

    def plot_2d(
        self,
        mode: str = "lines",
        renderer: str | None = None,
        **opts,
    ) -> None:
        """
        return a plotly Figure object representing the model.

        Parameters
        ----------
        mode: str
            determine the style of the plot. "lines" defines a line plot,
            "scatter" denotes a scatter plot.

        renderer: str | None
            the renderer to be used for showing the figure i.e. None (browser)
            or "notebook" for notebooks.

        **opts: object
            list of keyworded options passed to the plotly.express function
            used to generate the figure.
        """

        # get the data
        pdata = self.to_long_dataframe()
        cols = [f"{x} ({y})" for x, y in zip(pdata.DIMENSION, pdata.UNIT)]
        pdata.loc[pdata.index, "DIMENSION"] = cols
        pdata = pdata.drop("UNIT", axis=1)

        # plot the figure
        if not isinstance(mode, str):
            raise TypeError(f"mode must be a {str} instance.")
        if mode == "lines":
            fun = px.line
        elif mode == "scatter":
            fun = px.scatter
        else:
            raise ValueError("mode must be a 'lines' or 'scatter'.")
        fig = fun(
            data_frame=pdata,
            x="TIME (s)",
            y="VALUE",
            color="DIMENSION",
            facet_col="PROPERTY",
            facet_row="NAME",
            **opts,
        )

        # update the layout
        fig.for_each_yaxis(lambda a: a.update(text=""))
        fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))

        # show
        fig.show(renderer=renderer)

    def plot_3d(self, vertical_axis: str = "Y") -> None:
        """
        return a 3D representation of the model.

        Parameters
        ----------
        vertical_axis: str
            the dimension denoting the vertical axis.
        """
        if not isinstance(vertical_axis, str):
            raise TypeError(f"{vertical_axis} must be a {str} instance.")
        if not vertical_axis in self.dimensions:
            raise ValueError(f"{vertical_axis} not found in {self.dimensions}")

        # highdpi scaling
        high_dpi = qtcore.Qt.AA_EnableHighDpiScaling
        widgets.QApplication.setAttribute(high_dpi, True)
        high_pixmaps = qtcore.Qt.AA_UseHighDpiPixmaps
        widgets.QApplication.setAttribute(high_pixmaps, True)

        # app generation
        app = widgets.QApplication(sys.argv)
        widget = Model3DWidget(self, vertical_axis=vertical_axis)
        widget.show()
        app.exec_()

    def _validate(self, obj: object) -> None:
        """
        check whether the input object can be integrated within the model.

        Parameters
        ----------
        obj: object
            Any python object. In order to be a valid element for the model,
            the object must be a pandas dataframe having:
            - numerical-only values
            - simple numerical index
            - MultiIndex columns with structure: OBJECT_NAME, OBJECT_TYPE,
            PROPERTY_NAME, PROPERTY_UNIT, DIMENSION_NAME
        """

        # setup the error message
        err = f"{obj} must be a a {pd.DataFrame} having:\n"
        err += "\t- numerical-only values\n"
        err += "\t- simple numerical index\n"
        err += "\t- MultiIndex columns with no empty elements and structure:\n"
        err += "\t\tOBJECT_NAME\n"
        err += "\t\tOBJECT_TYPE\n"
        err += "\t\tPROPERTY_NAME\n"
        err += "\t\tPROPERTY_UNIT\n"
        err += "\t\tDIMENSION_NAME\n"

        # set the numpy accepted dtypes
        dtypes = set("if")

        # check the object validity
        if not isinstance(obj, (Model3D, pd.DataFrame)):
            raise TypeError(err)
        index = obj.index.to_numpy()
        if not index.dtype.kind in dtypes:
            raise TypeError(err)
        values = obj.values
        if not values.dtype.kind in dtypes:
            raise TypeError(err)
        columns = pd.DataFrame(obj).columns.to_frame()
        if columns.shape[0] != 5:
            raise TypeError(err)
        if columns.isna().any(1).any(0):
            raise TypeError(err)

        # check for duplicates
        old_cols = self._data.columns.to_numpy()
        for col in columns.T.values:
            if np.any(np.where(np.all(col in old_cols, 0))[0]):
                warnings.warn(f"duplicated {col}.")
