# SENSORS MODULE


#! IMPORTS


from .geometry import *
from plotly.subplots import make_subplots
import plotly.express.colors as pcol
import plotly.graph_objects as go
import numpy as np
import pandas as pd
import warnings


#! CLASSES


class Sensor:
    """
    general class containing just methods defining that the GeometricObject is
    also a Sensor.
    """

    def __init__(self):
        pass


class Marker3D(Point, Sensor):
    """
    generate a 3D object reflecting a marker collected over time in a 3D space.

    Parameters
    ----------
    coordinates: Point, pandas.DataFrame, numpy.ndarray, list
        the amplitude of the vector.

    index: arraylike
        the index for the samples of the sensor.

    unit: str
        the unit of measurement of the sensors's coordinates.
    """

    def __init__(self, coordinates, index=None, unit="m"):
        amp = self._get_data(
            data=coordinates,
            index=index,
            columns=["X", "Y", "Z"],
            unit=unit,
        )
        assert amp.ndim == 3, "a 3D dataset must be provided."
        super(Marker3D, self).__init__(coordinates=amp, unit=unit)


class ForcePlatform3D(GeometricObject, Sensor):
    """
    generate a 3D object reflecting a force platform whose data have been
    collected over time in a 3D space.

    Parameters
    ----------
    force: Vector, UnitDataFrame, Point, pandas.DataFrame, numpy.ndarray, list
        the amplitude of the force vector.

    moment: Vector, UnitDataFrame, Point, pandas.DataFrame, numpy.ndarray, list
        the amplitude of the moments vector.

    origin: UnitDataFrame, Point, pandas.DataFrame, numpy.ndarray, list
        the origin of the force and moments vectors.

    index: arraylike | None
        the index for the input data.

    force_unit: str
        the unit of measurement of the force data.

    moment_unit: str
        the unit of measurement of the moment data.

    origin_unit: str
        the unit of measurement of the origin data.
    """

    @property
    def force_vector(self):
        """
        return the force vector measured by the ForcePlaftorm.
        """
        return Vector(amplitude=self._force, origin=self._origin)

    @property
    def moment_vector(self):
        """
        return the moments vector measured by the ForcePlaftorm.
        """
        return Vector(amplitude=self._moment, origin=self._origin)

    def __init__(
        self,
        force,
        moment,
        origin,
        index=None,
        force_unit="N",
        moment_unit="Nm",
        origin_unit="m",
    ):
        # handle the case force and moment are vectors
        if isinstance(force, Vector) or isinstance(moment, Vector):
            txt = "force and moment vectors must match and have the same "
            txt += "origin."
            assert isinstance(force, Vector), txt
            assert isinstance(moment, Vector), txt
            assert force.matches(moment, strict=False), txt
            assert np.sum((force.origin - moment.origin).values) == 0, txt
            frz = force.amplitude
            mnt = moment.amplitude
            ori = force.origin
        else:
            frz = force.coordinates if isinstance(force, Point) else force
            mnt = moment.coordinates if isinstance(moment, Point) else moment
            ori = origin.coordinates if isinstance(origin, Point) else origin

        # check the size
        txt = "only 3D data can be provided."
        assert frz.ndim == 3, txt
        assert mnt.ndim == 3, txt
        assert ori.ndim == 3, txt
        frz = self._get_data(
            data=frz,
            index=index,
            columns=["X", "Y", "Z"],
            unit=force_unit,
        )
        mnt = self._get_data(
            data=mnt,
            index=index,
            columns=["X", "Y", "Z"],
            unit=moment_unit,
        )
        ori = self._get_data(
            data=ori,
            index=index,
            columns=["X", "Y", "Z"],
            unit=origin_unit,
        )

        # build the object
        super(ForcePlatform3D, self).__init__(
            force=frz,
            moment=mnt,
            origin=ori,
        )


class Link3D(Segment, Sensor):
    """
    Generate an object reflecting a dimension-less vector in a n-dimensional
    space.

    Parameters
    ----------

    p0, p1: Point, UnitDataFrame, pandas.DataFrame, numpy.ndarray, list
        the first and second points of the segment.

    index: arraylike
        the index for the samples of the sensor.

    unit: str
        the unit of measurement of the sensors.
    """

    def __init__(self, p0, p1, index=None, unit=""):
        v0 = p0.coordinates if isinstance(p0, Point) else p0
        assert v0.ndim == 3, "a 3D dataset must be provided."
        v0 = self._get_data(v0, index, ["X", "Y", "Z"], unit)
        v1 = p1.coordinates if isinstance(p1, Point) else p1
        assert v1.ndim == 3, "a 3D dataset must be provided."
        v1 = self._get_data(v1, index, ["X", "Y", "Z"], unit)
        super(Link3D, self).__init__(p0=v0, p1=v1)


class EmgSensor(GeometricMathObject, Sensor):
    """
    Generate a n-channels EMG sensor instance.

    Parameters
    ----------
    amplitude: UnitDataFrame, pandas.DataFrame, numpy.ndarray, list
        the amplitude of the EMG signal(s).

    index: arraylike
        the index for the samples of the sensor.

    unit: str
        the unit of measurement of the sensors.
    """

    def __init__(self, amplitude, index=None, unit=""):
        """
        constructor
        """
        if isinstance(amplitude, list):
            amp = np.array(list)
        else:
            amp = amplitude
        if isinstance(amp, np.ndarray):
            if amp.ndim == 1:
                amp = np.atleast_2d(amp).T
            elif amp.ndim > 2:
                raise ValueError("amplitude must be a 1D or 2D dataset.")
            cols = ["Channel{}".format(i + 1) for i in range(amp.shape[1])]
            df = self._get_data(amp, index, cols)
        elif isinstance(amp, (pd.DataFrame, UnitDataFrame)):
            df = self._get_data(amp)
        super(EmgSensor, self).__init__(amplitude=df, unit=unit)

    def _math_value(self, obj, transpose: bool = False) -> np.ndarray:
        """
        Parameters
        ----------
        obj: int, float, np.ndarray, UnitDataFrame, pd.DataFrame, Vector
            the second object included in the math operation.

        transpose: bool (optional, default=False)
            if True, the transpose of obj is checked. Otherwise the
            obj data is controlled as is.

        Returns
        -------
        val: np.ndarray
            the value to be used for the math operation.
        """
        if isinstance(obj, EmgSensor):
            val = obj.amplitude
        else:
            val = obj
        return super(EmgSensor, self)._math_value(val, transpose)

    def __add__(self, obj):
        """
        addition.
        """
        return EmgSensor(amplitude=self.amplitude + self._math_value(obj))

    def __sub__(self, obj):
        """
        subtraction.
        """
        return EmgSensor(amplitude=self.amplitude - self._math_value(obj))

    def __mul__(self, obj):
        """
        multiplication
        """
        return EmgSensor(amplitude=self.amplitude * self._math_value(obj))

    def __truediv__(self, obj):
        """
        division
        """
        return EmgSensor(amplitude=self.amplitude / self._math_value(obj))

    def __floordiv__(self, obj):
        """
        floor division
        """
        return EmgSensor(amplitude=self.amplitude // self._math_value(obj))

    def __pow__(self, obj):
        """
        power elevation (** operator).
        """
        return EmgSensor(amplitude=self.amplitude ** self._math_value(obj))

    def __mod__(self, obj):
        """
        module (% operator).
        """
        return EmgSensor(amplitude=self.amplitude % self._math_value(obj))

    def __matmul__(self, obj):
        """
        matrix multiplication (@ operator).
        """
        val = self._math_value(obj, transpose=True)
        return EmgSensor(amplitude=self.amplitude @ val)


class Imu3D(GeometricMathObject, Sensor):
    """
    generate a 3D object reflecting an Inertial Measurement Unit
    whose data have been collected over time in a 3D space.

    Parameters
    ----------
    accelerometer: Any of the valid data input (see below)
        the amplitude of the accelerations.

    gyroscope: Any of the valid data input (see below)
        the amplitude of angular velocities.

    magnetometer: Any of the valid data input (see below)
        the amplitude of the magnetic field readings.

    index: arraylike
        the index for the samples of the sensor.

    accelerometer_unit: str
        the unit of measurement of the accelerometer.

    gyroscope_unit: str
        the unit of measurement of the gyroscope.

    magnetometer_unit: str
        the unit of measurement of the magnetometer.

    Valid inputs
    ------------
    UnitDataFrame, Point, pandas.DataFrame, numpy.ndarray, list, None
    """

    def __init__(
        self,
        accelerometer,
        gyroscope,
        magnetometer,
        index=None,
        accelerometer_unit="m/s^2",
        gyroscope_unit="rad/s",
        magnetometer_unit="nT",
    ):
        attrs = {}
        txt = "only 3D data can be provided."
        if accelerometer is not None:
            if isinstance(accelerometer, Point):
                attrs["accelerometer"] = accelerometer.coordinates
                assert attrs["accelerometer"].ndim == 3, txt
            else:
                attrs["accelerometer"] = self._get_data(
                    data=accelerometer,
                    index=index,
                    coordinates=["X", "Y", "Z"],
                    unit=accelerometer_unit,
                )
        if gyroscope is not None:
            if isinstance(gyroscope, Point):
                attrs["gyroscope"] = gyroscope.coordinates
                assert attrs["gyroscope"].ndim == 3, txt
            else:
                attrs["gyroscope"] = self._get_data(
                    data=gyroscope,
                    index=index,
                    coordinates=["X", "Y", "Z"],
                    unit=gyroscope_unit,
                )
        if magnetometer is not None:
            if isinstance(magnetometer, Point):
                attrs["magnetometer"] = magnetometer.coordinates
                assert attrs["magnetometer"].ndim == 3, txt
            else:
                attrs["magnetometer"] = self._get_data(
                    data=magnetometer,
                    index=index,
                    coordinates=["X", "Y", "Z"],
                    unit=magnetometer_unit,
                )

        # build the object
        super(Imu3D, self).__init__(**attrs)


class Model3D:
    """
    generic class used as interface for the implementation of
    specific 3D models.

    Parameters
    ----------
    sensors: keyworded arguments
        The list of arguments containing the data of the object.
        The key of the arguments will be used as attributes of the object.
        The values of each key must be an instance of the Sensor class.
    """

    # list of sensors of the class which contain relevant data.
    # NOTE THESE ATTRIBUTES ARE CONSTANTS THAT SHOULD NOT BE MODIFIED
    _sensors = []

    def __init__(self, **sensors):
        """
        class constructor.
        """
        self._sensors = []

        # populate the object with the input sensors
        for attr, value in sensors.items():
            self.append(obj=value, name=attr)

    def __str__(self) -> str:
        """
        convert self to a string.
        """
        return self.pivot().__str__()

    def append(self, obj, name):
        """
        append an object to self and store it appropriately.

        Parameters
        ----------
        obj: GeometricObject instance
            the object to be stored.

        name: str
            the label denoting the object.
        """

        # check the input data
        txt = "obj must be a Sensor instance."
        assert isinstance(obj, Sensor), txt
        assert isinstance(name, str), "name must be a string."

        # ensure the object will be stored according to its class
        cls = obj.__class__.__name__
        if not cls in self._sensors:
            self._sensors += [cls]
            setattr(self, cls, {})

        # check if another object with the same name exists
        if any([i == name for i in getattr(self, cls)]):
            txt = "{} already existing in {}. The old instance has been "
            txt += "replaced"
            warnings.warn(txt.format(name, cls))

        # store the sensor
        getattr(self, cls).update({name: obj})

    def pivot(self) -> pd.DataFrame:
        """
        generate a wide dataframe object containing both origin and
        amplitudes.
        """
        col = ["Sensor", "Label", "Source", "Dimension"]
        df = self.stack().pivot("Time", col)
        df.columns = pd.Index([i[1:] for i in df.columns])
        return df

    def _colors(self, as_subplots=True):
        """
        colors used in the plot generation.
        """
        if as_subplots:
            nrows = len(self.EmgSensor)
        else:
            nrows = 1
        # set the colors
        palette = pcol.qualitative.Plotly + pcol.qualitative.D3
        emg_col = []
        for i in range(nrows):
            emg_col += [palette[3] if as_subplots else palette[(i + 3) % 20]]
        return {
            "Point": palette[0],
            "Vector": palette[1],
            "Segment": palette[2],
            "EmgSensor": emg_col,
        }

    def _make_figure(self, as_subplots=True):
        """
        return the background frame on which including the animated
        traces.

        Parameters
        ----------

        as_subplots: bool (default=True)
            if EmgSensor data is available, should they be separated
            by channel into single line plots?

        Returns
        -------
        fig: plotly.FigureWidget
            the figure representing the required frame.
        """

        # generate the figure grid
        plot_emg = any([i == EmgSensor.__name__ for i in self.sensors])
        if plot_emg:
            if as_subplots:
                sub_titles = ["3D View"] + [i for i in self.EmgSensor]
                nrows = len(self.EmgSensor)
            else:
                sub_titles = ["3D View", "EMG"]
                nrows = 1
            ncols = 2
            specs = [[{"type": "scene"}, {}]]
            specs += [[None, {}] for _ in range(nrows - 1)]
            col_widths = [0.7, 0.3]
        else:
            nrows = 1
            ncols = 1
            specs = [[{"type": "scene"}]]
            col_widths = [1]
            sub_titles = ["3D View"]
        fig = make_subplots(
            rows=nrows,
            cols=ncols,
            column_widths=col_widths,
            subplot_titles=sub_titles,
            specs=specs,
            shared_xaxes=True,
        )

        # populate the emg plot(s) with the whole signals
        colors = self._colors(as_subplots)
        if plot_emg:
            for i, label in enumerate(self.EmgSensor):
                obj = self.EmgSensor[label]
                t = obj.index
                y = obj.amplitude.values.flatten()
                fig.add_trace(
                    row=i + 1,
                    col=ncols,
                    trace=go.Scatter(
                        x=t,
                        y=y,
                        mode="lines",
                        name=label,
                        legendgroup="EMG",
                        legendgrouptitle_text="EMG",
                        showlegend=not as_subplots,
                        line=dict(
                            color=colors["EmgSensor"][i],
                            dash="solid",
                            width=2,
                        ),
                    ),
                )

                # plot an empty vertical line
                fig.add_trace(
                    row=i + 1,
                    col=ncols,
                    trace=go.Scatter(
                        x=[],
                        y=[],
                        name="_" + label,
                        mode="lines",
                        showlegend=False,
                        line=dict(
                            color="grey",
                            dash="dash",
                            width=1,
                        ),
                    ),
                )

        # update the axes such as the x-axis of the EMG
        # subplots will be shared with each-other
        fig.update_xaxes(matches="x1")

        # add the markers
        plot_markers = any([i == Marker3D.__name__ for i in self.sensors])
        if plot_markers:
            for i, label in enumerate(self.Marker3D):
                fig.add_trace(
                    row=1,
                    col=1,
                    trace=go.Scatter3d(
                        x=[],
                        y=[],
                        z=[],
                        mode="markers",
                        name=label,
                        text=label,
                        legendgroup="Marker3D",
                        legendgrouptitle_text="Marker3D",
                        showlegend=False,
                        marker=dict(color=colors["Point"], size=8),
                    ),
                )

        # add the links
        plot_links = any([i == Link3D.__name__ for i in self.sensors])
        if plot_links:
            for i, label in enumerate(self.Link3D):
                fig.add_trace(
                    row=1,
                    col=1,
                    trace=go.Scatter3d(
                        x=[],
                        y=[],
                        z=[],
                        mode="lines",
                        name="_" + label,
                        showlegend=False,
                        marker=dict(color=colors["Segment"], size=4),
                    ),
                )

        # add the forces
        plot_fp = any([i == ForcePlatform3D.__name__ for i in self.sensors])
        if plot_fp:
            for i, label in enumerate(self.ForcePlatform3D):
                fig.add_traces(
                    rows=1,
                    cols=1,
                    data=[
                        go.Scatter3d(
                            x=[],
                            y=[],
                            z=[],
                            opacity=0.5,
                            mode="lines",
                            name=label,
                            text=label,
                            legendgroup="Force3D",
                            legendgrouptitle_text="Force3D",
                            showlegend=False,
                            marker=dict(color=colors["Vector"], size=6),
                        ),
                        go.Cone(
                            x=[],
                            y=[],
                            z=[],
                            u=[],
                            v=[],
                            w=[],
                            opacity=0.5,
                            sizemode="absolute",
                            sizeref=2,
                            anchor="tip",
                            name=label,
                            text=label,
                            colorscale=[
                                [0, colors["Vector"]],
                                [1, colors["Vector"]],
                            ],
                            legendgroup="Force3D",
                            legendgrouptitle_text="Force3D",
                            showlegend=False,
                        ),
                    ],
                )

        return fig

    def _make_frame(self, index, as_subplots=True):
        """
        return a plotly figure of the frame corresponding to the
        provided index.

        Parameters
        ----------
        index: float, int
            the time index of which the frame is required.

        as_subplots: bool (default=True)
            if EmgSensor data is available, should they be separated
            by channel into single line plots?

        Returns
        -------
        fig: plotly.FigureWidget
            the figure representing the required frame.
        """

        # generate the figure grid
        plot_emg = any([i == EmgSensor.__name__ for i in self.sensors])
        if plot_emg:
            if as_subplots:
                sub_titles = ["3D View"] + [i for i in self.EmgSensor]
                nrows = len(self.EmgSensor)
            else:
                sub_titles = ["3D View", "EMG"]
                nrows = 1
            ncols = 2
            specs = [[{"type": "scene"}, {}]]
            specs += [[None, {}] for _ in range(nrows - 1)]
            col_widths = [0.7, 0.3]
        else:
            nrows = 1
            ncols = 1
            specs = [[{"type": "scene"}]]
            col_widths = [1]
            sub_titles = ["3D View"]
        fig = make_subplots(
            rows=nrows,
            cols=ncols,
            column_widths=col_widths,
            subplot_titles=sub_titles,
            specs=specs,
        )

        # set the colors
        palette = pcol.qualitative.Plotly + pcol.qualitative.D3
        emg_col = []
        for i in range(nrows):
            emg_col += [palette[3] if as_subplots else palette[(i + 3) % 20]]
        colors = {
            "Point": palette[0],
            "Vector": palette[1],
            "Segment": palette[2],
            "EmgSensor": emg_col,
        }

        # populate the emg plot(s)
        # include the whole data and a vertical dashed bar
        # at the time instant corresponding to index.
        for i, label in enumerate(self.EmgSensor):
            obj = self.EmgSensor[label]
            t = obj.index
            y = obj.amplitude.values.flatten()
            idx = np.where(t == index)[0]
            y_index = y[idx]
            fig.add_traces(
                rows=i + 1,
                cols=ncols,
                data=[
                    go.Scatter(
                        x=t,
                        y=y,
                        mode="lines",
                        name=label,
                        legendgroup="EMG",
                        legendgrouptitle_text="EMG",
                        showlegend=not as_subplots,
                        line=dict(
                            color=colors["EmgSensor"][i],
                            dash="solid",
                            width=4,
                        ),
                    ),
                    go.Scatter(
                        x=[index, index],
                        y=[y_index, y_index],
                        mode="lines",
                        showlegend=False,
                        line=dict(
                            color="grey",
                            dash="dash",
                            width=2,
                        ),
                    ),
                ],
            )

        # update the axes such as the x-axis of the EMG
        # subplots will be shared with each-other
        fig.update_xaxes(matches="x1")

        # add the markers
        plot_markers = any([i == Marker3D.__name__ for i in self.sensors])
        if plot_markers:
            for i, label in enumerate(self.Marker3D):
                obj = self.Marker3D[label]
                o = obj[index].dropna()
                if o.nsamp > 0:
                    x, y, z = o.coordinates.values.flatten()
                else:
                    x, y, z = ([], [], [])
                fig.add_trace(
                    row=1,
                    col=1,
                    trace=go.Scatter3d(
                        x=list(x),
                        y=list(y),
                        z=list(z),
                        mode="markers",
                        name=label,
                        text=label,
                        legendgroup="Marker3D",
                        legendgrouptitle_text="Marker3D",
                        showlegend=False,
                        marker=dict(color=colors["Point"], size=8),
                    ),
                )

        # add the links
        plot_links = any([i == Link3D.__name__ for i in self.sensors])
        if plot_links:
            for i, label in enumerate(self.Link3D):
                obj = self.Link3D[label]
                o = obj[index].dropna()
                if o.nsamp > 0:
                    x0, y0, z0 = o.p0.values.flatten()
                    x1, y1, z1 = o.p1.values.flatten()
                    x = [x0, x1]
                    y = [y0, y1]
                    z = [z0, z1]
                else:
                    x, y, z = ([], [], [])
                fig.add_trace(
                    row=1,
                    col=1,
                    trace=go.Scatter3d(
                        x=x,
                        y=y,
                        z=z,
                        mode="lines",
                        name=label,
                        showlegend=False,
                        marker=dict(color=colors["Segment"], size=4),
                    ),
                )

        # add the forces
        plot_fp = any([i == ForcePlatform3D.__name__ for i in self.sensors])
        if plot_fp:
            for i, label in enumerate(self.ForcePlatform3D):
                obj = self.ForcePlatform3D[label].force_vector()
                o = obj[index].dropna()
                if o.nsamp > 0:
                    p0 = o.origin
                    p2 = p0 + o.amplitude.values
                    p1 = Segment(p0, p2).point_at(0.75, True)
                    v0 = p0.values.flatten()
                    v1 = p1.values.flatten()
                    v2 = p2.values.flatten()
                    x_line, y_line, z_line = [[i, v] for i, v in zip(v0, v1)]
                    cones_coords = [[i, v] for i, v in zip(v2, v2 - v1)]
                    x_cone, y_cone, z_cone = cones_coords

                else:
                    x_line, y_line, z_line = [[], [], []]
                    x_cone, y_cone, z_cone = [[], [], []]
                fig.add_traces(
                    rows=1,
                    cols=1,
                    data=[
                        go.Scatter3d(
                            x=x_line,
                            y=y_line,
                            z=z_line,
                            opacity=0.5,
                            mode="lines",
                            name=label,
                            text=label,
                            legendgroup="Force3D",
                            legendgrouptitle_text="Force3D",
                            showlegend=False,
                            marker=dict(color=colors["Vector"], size=6),
                        ),
                        go.Cone(
                            x=x_cone[0],
                            y=y_cone[0],
                            z=z_cone[0],
                            u=x_cone[1],
                            v=y_cone[2],
                            w=z_cone[3],
                            opacity=0.5,
                            sizemode="absolute",
                            sizeref=2,
                            anchor="tip",
                            name=label,
                            text=label,
                            colorscale=[
                                [0, colors["Vector"]],
                                [1, colors["Vector"]],
                            ],
                            legendgroup="Force3D",
                            legendgrouptitle_text="Force3D",
                            showlegend=False,
                        ),
                    ],
                )

        return fig

    def plot(
        self,
        as_subplots=True,
        show: bool = True,
        width: int = 1280,
        height: int = 720,
    ):
        """
        generate a plotly plot representing the current object.

        Parameters
        ----------
        as_subplots: bool (default=True)
            if EmgSensor data is available, should they be separated
            by channel into single line plots?

        show: bool (default=True)
            if True the generated figure is immediately plotted.
            Otherwise the generated object is returned.

        width: int (default=1280)
            the width of the output figure in pixels

        height: int (default=720)
            the height of the output figure in pixels

        Returns
        -------
        None, if show = True. A plotly.Figure object, otherwise.
        """

        # get the background structure
        fig = self._make_figure(as_subplots)

        # update the layout
        fig.update_layout(
            width=width,
            height=height,
            template="simple_white",
        )

        # get the adjustable traces indices
        traces = []
        for i, v in enumerate(fig.data):
            if v.name[0] == "_" or v.legendgroup in ["Force3D", "Marker3D"]:
                traces += [i]

        fig.show()

        # get the samples of the model
        idx = self.pivot().index.to_numpy()

        # get the frames corresponding to each sample
        frames = [self._make_frame(i, as_subplots) for i in idx]

    def describe(self, percentiles: list = []) -> pd.DataFrame:
        """
        provide descriptive statistics about the parameters in df.

        Parameters
        ----------
        percentiles: list
            a list of values in the [0, 1] range defining the desired
            percentiles to be calculated.

        Returns
        -------
        df: pd.DataFrame
            a pandas.DataFrame with the object descriptive statistics.
        """
        grp = self.stack().drop("Time", axis=1)
        grp = grp.groupby(["Sensor", "Label", "Source", "Dimension"])
        df = grp.describe(percentiles=percentiles)
        df.columns = pd.Index([i[1] for i in df.columns])
        return df

    def stack(self) -> pd.DataFrame:
        """
        stack the object as a long format DataFrame.
        """
        out = []
        for sns in self._sensors:
            for lbl, obj in getattr(self, sns).items():
                df = obj.stack()
                df.insert(0, "Label", np.tile(lbl, df.shape[0]))
                df.insert(0, "Sensor", np.tile(sns, df.shape[0]))
                out += [df]
        return pd.concat(out, axis=0, ignore_index=True)

    def copy(self):
        """
        make a copy of the object
        """
        return self.unstack(self.stack())

    @classmethod
    def unstack(cls, df: pd.DataFrame):
        """
        convert a long format DataFrame into an instance of the object.

        Parameters
        ----------
        df: pandas.DataFrame
            a pandas.DataFrame sorted as it would be generated by the
            .stack() method.

        Returns
        -------
        obj: GeometricObject
            the instance resulting from the dataframe reading.
        """
        out = cls()
        root = ["Type", "Sensor"]
        types = np.unique(df[root].values.astype(str), axis=0)
        for typ, sns in types:
            sub = df.loc[df[root].isin([sns, typ]).all(1)]
            out.append(obj=eval(typ).unstack(sub), name=sns)
        return out

    @property
    def sensors(self):
        """
        return the sensors of this instance.
        """
        return self._sensors
