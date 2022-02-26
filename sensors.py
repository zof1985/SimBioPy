# SENSORS MODULE


#! IMPORTS

import os
import warnings
import numpy as np
import pandas as pd
import PySide2.QtWidgets as qtw
import PySide2.QtCore as qtc
import PySide2.QtGui as qtg
import matplotlib.pyplot as pl
from matplotlib.gridspec import GridSpec
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as Canvas
from matplotlib import use as matplotlib_use
from mpl_toolkits import mplot3d
from .geometry import *

matplotlib_use("Qt5Agg")


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
        # populate the object with the input sensors
        self._sensors = []
        self.append(**sensors)

    def __str__(self) -> str:
        """
        convert self to a string.
        """
        return self.pivot().__str__()

    def _appender(self, obj, name):
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

    def append(self, **objs):
        """
        append named objects to this model.

        Parameters
        ----------
        objs: keyword-named sensor
            any object being instance of the Sensor class.
        """
        for i, v in objs.items():
            self._appender(obj=v, name=i)

    def pivot(self) -> pd.DataFrame:
        """
        generate a wide dataframe object containing both origin and
        amplitudes.
        """
        col = ["Sensor", "Label", "Source", "Dimension"]
        df = self.stack().pivot("Time", col)
        df.columns = pd.Index([i[1:] for i in df.columns])
        return df

    def has_Marker3D(self):
        """
        check whether 3D markers are included in the model.
        """
        return any([i == Marker3D.__name__ for i in self.sensors])

    def has_ForcePlatform3D(self):
        """
        check whether 3D force platform are included in the model.
        """
        return any([i == ForcePlatform3D.__name__ for i in self.sensors])

    def has_EmgSensor(self):
        """
        check whether EMG sensors are included in the model.
        """
        return any([i == EmgSensor.__name__ for i in self.sensors])

    def has_Link3D(self):
        """
        check whether 3D Links are included in the model.
        """
        return any([i == Link3D.__name__ for i in self.sensors])

    def has_Imu3D(self):
        """
        check whether 3D Links are included in the model.
        """
        return any([i == Imu3D.__name__ for i in self.sensors])

    def copy(self):
        """
        make a copy of the current object.
        """
        obj = Model3D()
        for attr in self.sensors:
            for key, val in getattr(self, attr).items():
                obj.append(val, key)
        return obj

    def drop(self, sensor, inplace=False):
        """
        remove the current sensor from the object.

        Parameters
        ----------
        sensor: str
            the name of the sensor to be removed.

        inplace: bool
            if True the current object is edited. Otherwise an edited copy
            is returned

        Returns
        -------
        None if inplace is True, a Model3D object otherwise.
        """
        assert isinstance(sensor, str), "sensor must be a str object."
        assert isinstance(inplace, bool), "inplace must be a bool."
        if inplace:
            for i in self.sensors:
                if i == sensor:
                    delattr(self, i)
                    break
        else:
            obj = Model3D()
            for i in self.sensors:
                if i != sensor:
                    for key, value in getattr(self, sensor).items():
                        obj.append(value, key)
            return obj

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

    @property
    def index(self):
        """
        return the index of the objects containing by the model.
        """
        return self.pivot().index.to_numpy().astype(int)


class Model3DWidget(qtw.QWidget, Model3D):
    """
    renderer for a 3D Model.
    """

    # class objects
    canvas = None
    slider = None
    time_label = None
    play_button = None
    forward_button = None
    backward_button = None
    repeat_button = None

    # private variables
    _font_size = 10
    _timer = None
    _update_rate = 10  # msec
    _is_running = False
    _figure = None
    _axis3D = None
    _axisEMG = []
    _dpi = 300
    _data = None
    _actual_frame = None

    def __init__(self, parent=None, **sensors):
        """
        constructor
        """
        super(Model3DWidget, self).__init__(parent=parent, **sensors)

        # timer
        self._timer = qtc.QTimer()
        self._timer.timeout.connect(self._update_figure)

        # buttons
        self.backward_button = self._render_button(
            label=os.path.sep.join(["icons", "backward.ico"]),
            event_handler=self._backward_pressed,
        )
        self.play_button = self._render_button(
            label=os.path.sep.join(["icons", "play.ico"]),
            event_handler=self._play_pressed,
        )
        self.forward_button = self._render_button(
            label=os.path.sep.join(["icons", "forward.ico"]),
            event_handler=self._forward_pressed,
        )
        self.repeat_button = self._render_button(
            label=os.path.sep.join(["icons", "repeat.ico"]),
            event_handler=self._repeat_pressed,
        )

        # forward step button
        self.forward_button = qtw.QPushButton("▶▶")
        self.forward_button.setFixedHeight = 35
        self.forward_button.setFixedWidth = 35
        self.forward_button.setFont = qtg.QFont("Arial", self.font_size)
        self.forward_button.clicked.connect(self._forward_pressed)

        # slider
        self.slider = qtw.QSlider()
        self.slider.setMinimum = 0
        self.slider.setTickInterval(1)
        self.slider.valueChanged.connect(self._update_figure)

        # play/pause button
        self.play_button = qtw.QPushButton("▶")
        self.play_button.setFixedHeight = 35
        self.play_button.setFixedWidth = 35
        self.play_button.setFont = qtg.QFont("Arial", self.font_size)
        self.play_button.clicked.connect(self._play_pressed)

        # forward step button
        self.forward_button = qtw.QPushButton("▶▶")
        self.forward_button.setFixedHeight = 35
        self.forward_button.setFixedWidth = 35
        self.forward_button.setFont = qtg.QFont("Arial", self.font_size)
        self.forward_button.clicked.connect(self._forward_pressed)

        # commands widget
        commands_layout = qtw.QHBoxLayout()
        commands_layout.addWidget(self.backward_button)
        commands_layout.addWidget(self.play_button)
        commands_layout.addWidget(self.forward_button)
        commands_layout.addWidget(self.slider)
        commands_layout.addWidget(self.time_label)
        commands_layout.addWidget(self.repeat_button)
        commands_widget = qtw.QWidget()
        commands_widget.setLayout(commands_layout)

        # image pane
        self.canvas = Canvas(self._figure)

        # widget layout
        layout = qtw.QVBoxLayout()
        layout.addWidget(self.canvas)
        layout.addWidget(commands_widget)
        self.setLayout(layout)

        # generate the figure grid
        if self.has_EmgSensor():
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
            shared_xaxes=False,
        )

        # get the trace colors
        colors = self._colors(as_subplots)

        # get the traces
        traces = []
        data = self.stack()
        points = data.loc[data.isin(["Marker3D"]).any(1)]
        if points.shape[0] > 0:
            points = points.pivot(["Time", "Label"], "Dimension", "Amplitude")
            lbl = points.index.to_frame()["Label"].values.flatten()
            points.insert(0, "Text", lbl)
            points.columns = pd.Index([i.lower() for i in points.columns])
            times = points.index.to_frame()["Time"].values.flatten()
            points.index = pd.Index(times)

        emg = data.loc[data.isin(["EmgSensor"]).any(1)]
        if emg.shape[0] > 0:
            emg = emg.pivot("Time", "Label", "Amplitude")
            emg_min = pd.DataFrame(emg.min(0)).T
            emg_max = pd.DataFrame(emg.max(0)).T
            emg_data = {}
            for muscle in emg.columns:
                if as_subplots:
                    min_val = emg_min[muscle].values[0]
                    max_val = emg_max[muscle].values[0]
                else:
                    min_val = np.min(emg_min.values.flatten())
                    max_val = np.max(emg_max.values.flatten())
                df = emg[[muscle]]
                df.columns = pd.Index(["Amplitude"])
                df.insert(0, "Max", np.tile(min_val, df.shape[0]))
                df.insert(0, "Min", np.tile(max_val, df.shape[0]))
                emg_data[muscle] = df
            traces["EMG"] = emg_data

        segments = data.loc[data.isin(["Link3D"]).any(1)]
        if segments.shape[0] > 0:
            segments_data = {}
            for lbl in np.unique(segments["Label"].values.flatten()):
                df = segments.loc[segments.isin([lbl]).any(1)]
                df0 = df.loc[df.isin(["p0"]).any(1)]
                df0 = df0.pivot("Time", "Dimension", "Amplitude")
                df0.insert(0, "Source", np.tile("p0", df0.shape[0]))
                df1 = df.loc[df.isin(["p1"]).any(1)]
                df1 = df1.pivot("Time", "Dimension", "Amplitude")
                df1.insert(0, "Source", np.tile("p1", df0.shape[0]))
                segments_data[lbl] = pd.concat([df0, df1], axis=0)
            traces["Link3D"] = segments_data

        forces = data.loc[data.isin(["ForcePlatform3D"]).any(1)]
        if forces.shape[0] > 0:
            force_data = {}
            for lbl in np.unique(forces["Label"].values.flatten()):
                df = forces.loc[forces.isin([lbl]).any(1)]
                ori = df.loc[df.isin(["origin"]).any(1)]
                ori = ori.pivot("Time", "Dimension", "Amplitude")
                ori.insert(0, "Source", np.tile("Origin", ori.shape[0]))
                frz = df.loc[df.isin(["force"]).any(1)] * force_scaler
                frz = frz.pivot("Time", "Dimension", "Amplitude")
                frz.insert(0, "Source", np.tile("Force", frz.shape[0]))
                force_data[lbl] = pd.concat([ori, frz], axis=0)
            traces["ForcePlatform3D"] = force_data

        # populate the emg plot(s) with the whole signals
        if emg.shape[0] > 0:
            for i, muscle in enumerate(emg_data):
                t = emg_data[muscle].index.to_numpy()
                y = emg_data[muscle]["Amplitude"].values.flatten()
                fig.add_trace(
                    row=1 + (i if as_subplots else 0),
                    col=ncols,
                    trace=go.Scatter(
                        x=t,
                        y=y,
                        mode="lines",
                        name=muscle,
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

        # update the layout
        fig.update_layout(
            width=width,
            height=height,
            template="simple_white",
            updatemenus=[
                {
                    "buttons": [
                        self._animation_button("PLAY", 2),
                        self._animation_button("PAUSE", 0),
                    ],
                    "direction": "left",
                    "pad": {"r": 10, "t": 87},
                    "showactive": False,
                    "type": "buttons",
                    "x": 0.1,
                    "xanchor": "right",
                    "y": 0,
                    "yanchor": "top",
                }
            ],
            sliders=[
                self._animation_slider(
                    steps=self.index,
                    label="TIME (msec): ",
                    duration=2,
                ),
            ],
        )

        # split the data in frames
        def get_sample(obj, i):
            if isinstance(obj, dict):
                return {j: get_sample(k, i) for j, k in obj.items()}
            new = obj.loc[i]
            if isinstance(new, pd.Series):
                new = pd.DataFrame(new).T
            return new

        frames = [get_sample(traces, i) for i in times[:100]]

        # get the EMG amplitude ranges
        emg_rng = {}
        if self.has_EmgSensor():
            for lbl, obj in self.EmgSensor.items():
                min_val = np.min(obj.amplitude.values.flatten())
                max_val = np.max(obj.amplitude.values.flatten())
                emg_rng[lbl] = [min_val, max_val]
            if not as_subplots:
                min_val = np.min([v[0] for v in emg_rng.values()])
                max_val = np.max([v[1] for v in emg_rng.values()])
                for lbl, obj in self.EmgSensor:
                    emg_rng[lbl] = [min_val, max_val]

        # get the valid time indices
        times = self.pivot()
        times = times.loc[times.notna().all(1)].index.to_numpy()

        # add the first frame to the static traces
        frame0 = self._plotly_traces(
            time=times[0],
            emg_ranges=emg_rng,
            as_subplots=as_subplots,
            force_scaler=force_scaler,
        )
        for t in frame0:
            dct = {"trace": t["trace"], "row": t["row"], "col": t["col"]}
            fig.add_trace(**dct)

        # add the frames to be animated
        def _frames(i):
            frame_data = self._plotly_traces(
                time=i,
                emg_ranges=emg_rng,
                as_subplots=as_subplots,
                force_scaler=force_scaler,
            )
            data = [j["trace"] for j in frame_data]
            traces = [j["n"] for j in frame_data]
            return go.Frame(data=data, traces=traces, name=str(i))

        self.frames = [_frames(i) for i in times]

        # return the figure or plot it
        if show:
            fig.show()
        else:
            return fig

    def append(self, **objs):
        """
        append named objects to this model.

        Parameters
        ----------
        objs: keyword-named sensor
            any object being instance of the Sensor class.
        """
        super(Model3DWidget, self).append(**objs)
        self._update_data()

    def _update_data(self):
        """
        private function used to update the data when new sensors are
        appended.
        """
        data = self.stack()
        times, idx = np.unique(
            ar=data["Time"].values.flatten(),
            return_index=True,
        )
        frames = np.arange(len(idx))
        for i, v in enumerate(idx):
            frames[v] = i
        data.insert(0, "FRAME", frames)
        self._data = data.groupby(["FRAME"])

        # update the slider
        self.slider.setMaximum = len(frames) - 1
        self.slider.setValue = 0

        # make the figure
        rows = len(self.EmgSensor) if self.has_EmgSensor() else 1
        grid = GridSpec(rows, 3)
        self._figure = pl.Figure(dpi=self._dpi)

        # populate the EMG data
        self._axisEMG = []
        if self.has_EmgSensor():
            for i, s in enumerate(self.EmgSensor):

                # plot the whole EMG signal
                time = s.amplitude.index.to_numpy()
                amplitude = s.amplitude.values.flatten()
                ax = self._figure.add_subplot(grid[i, 2])
                ax.plot(time, amplitude)
                ax.set_title(i)

                # share the x axis
                if i > 0:
                    ax.get_shared_x_axes().join(self._axisEMG[0], ax)
                    ax.set_xticklabels([])

                # store the axis
                self._axisEMG += [ax]

        # add the 3D model projection
        self._axis3D = self._figure.add_subplot(grid[:, :2], projection="3d")

        # render the first frame
        self._plot_frame(0)

    def _render_button(self, label, event_handler):
        """
        private method used to generate valid buttons for this widget.

        Parameters
        ----------
        label: str
            the label of the button

        event_handler: function
            the method to be passed as event handler for having the
            button being pressed.

        Returns
        -------
        obj: qtw.QPushButton
            a novel PushButton object.
        """
        button = qtw.QPushButton(label)
        button.setFixedHeight = 35
        button.setFixedWidth = 35
        button.setFont = qtg.QFont("Arial", self.font_size)
        button.clicked.connect(event_handler)
        return button

    def is_running(self):
        """
        check if the player is running.
        """
        return self._is_running

    def _play_pressed(self):
        """
        method handling the play button press events.
        """
        if self.is_running():
            self.play_button.setIcon(os.path.sep.join(["icons", "backward.ico"]))
            self.play_button.setText("❚❚")
            self._timer.start(self._update_rate)
        else:
            self._timer.stop()
            self.play_button.setText("▶")

    def _get_frame(self, frame):
        """
        return the long-form dataframe containing the data of the model
        at the required frame.

        Parameters
        ----------
        frame: int
            the index of the required frame.

        Returns
        -------
        df: pd.DataFrame
            the dataframe containing the data of the model at the required
            frame.
        """
        return self._data.get_group(frame)
