# SENSORS MODULE


#! IMPORTS

import os
from threading import Thread
import warnings
import numpy as np
import pandas as pd
import PySide2.QtWidgets as qtw
import PySide2.QtCore as qtc
import PySide2.QtGui as qtg
import matplotlib.pyplot as pl
from matplotlib.gridspec import GridSpec
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT
from matplotlib import use as matplotlib_use
from mpl_toolkits import mplot3d
from .geometry import *

# matplotlib options
matplotlib_use("Qt5Agg")
pl.rc("font", size=3)  # controls default text sizes
pl.rc("axes", titlesize=3)  # fontsize of the axes title
pl.rc("axes", labelsize=3)  # fontsize of the x and y labels
pl.rc("xtick", labelsize=3)  # fontsize of the x tick labels
pl.rc("ytick", labelsize=3)  # fontsize of the y tick labels
pl.rc("legend", fontsize=3)  # legend fontsize
pl.rc("figure", titlesize=3)  # fontsize of the figure title

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
        generate a wide dataframe object containing all the data.
        """
        out = []
        for sensor in self.sensors:
            for label, value in getattr(self, sensor).items():
                df = value.pivot()
                cols = ([sensor, label, *i] for i in df.columns)
                df.columns = pd.MultiIndex.from_tuples(cols)
                out += [df]
        return pd.concat(out, axis=1)

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


class Model3DWidget(qtw.QWidget):
    """
    renderer for a 3D Model.
    """

    # class objects
    model = None
    slider = None
    time_label = None
    play_button = None
    forward_button = None
    backward_button = None
    repeat_button = None

    # private variables
    _font_size = 12
    _button_size = 75
    _play_timer = None
    _update_rate = 10  # msec
    _is_running = False
    _model3D = None
    _figureEMG = None
    _axis3D = None
    _marker3D = {}
    _force3D = {}
    _link3D = {}
    _text3D = {}
    _force3D_sclr = 1
    _emg = []
    _dpi = 300
    _actual_frame = None
    _frames = None
    _threads = []

    def __init__(self, model, parent=None):
        """
        constructor
        """
        super(Model3DWidget, self).__init__(parent=parent)

        # store the model
        txt = "model must be a Model3D instance."
        assert isinstance(model, Model3D), txt
        self.model = model

        # path to the package folder
        self._path = os.path.sep.join([os.getcwd(), "simbiopy"])

        # get the time and frames
        idx = {"Sensor": ["ForcePlatform3D", "Marker3D", "Link3D"]}
        self._data = self.model.stack()
        idx = self._data.isin(idx).any(1).values.flatten()
        if len(idx) > 0:
            self._times = np.unique(self._data.loc[idx]["Time"].values.flatten())
        else:
            self._times = np.unique(self._data["Time"].values.flatten())
        self._actual_frame = 0
        frame = {
            "Marker3D": None,
            "ForcePlatform3D": None,
            "Link3D": None,
            "EmgSensor": None,
        }
        self.frames = {i: frame.copy() for i in self._times}

        # buttons
        self.backward_button = self._render_button(
            icon=os.path.sep.join([self._path, "icons", "backward.png"]),
            event_handler=self._backward_pressed,
        )
        self.play_button = self._render_button(
            icon=os.path.sep.join([self._path, "icons", "play.png"]),
            event_handler=self._play_pressed,
        )
        self.forward_button = self._render_button(
            icon=os.path.sep.join([self._path, "icons", "forward.png"]),
            event_handler=self._forward_pressed,
        )
        self.repeat_button = self._render_button(
            icon=os.path.sep.join([self._path, "icons", "repeat.png"]),
        )
        self.repeat_button.setCheckable(True)

        # slider
        self.slider = qtw.QSlider(qtc.Qt.Horizontal)
        self.slider.setValue(0)
        self.slider.setMinimum(0)
        self.slider.setMaximum(len(self.frames) - 1)
        self.slider.setTickInterval(1)
        self.slider.valueChanged.connect(self._slider_moved)

        # time label
        self.time_label = qtw.QLabel("00:00.000")
        self.time_label.setFont(qtg.QFont("Arial", self._font_size))
        self.time_label.setFixedHeight(35)
        self.time_label.setFixedWidth(100)
        self.time_label.setAlignment(qtc.Qt.AlignVCenter)

        # commands widget
        commands_layout = qtw.QHBoxLayout()
        commands_layout.addWidget(self.backward_button)
        commands_layout.addWidget(self.play_button)
        commands_layout.addWidget(self.forward_button)
        commands_layout.addWidget(self.repeat_button)
        commands_layout.addWidget(self.time_label)
        commands_layout.addWidget(self.slider)
        commands_widget = qtw.QWidget()
        commands_widget.setLayout(commands_layout)
        opacity_zero = qtw.QGraphicsOpacityEffect()
        opacity_zero.setOpacity(0)
        commands_widget.setGraphicsEffect(opacity_zero)

        # make the EMG pane
        if self.model.has_EmgSensor():

            # figure
            rows = len(self.model.EmgSensor)
            grid = GridSpec(rows, 1)
            self._figureEMG = pl.figure(dpi=self._dpi)
            self._figureEMG.tight_layout()
            self.canvasEMG = FigureCanvasQTAgg(self._figureEMG)

            # add the emg data
            self._emg = {}
            emg_axes = []
            for i, s in enumerate(self.model.EmgSensor):

                # plot the whole EMG signal
                ax = self._figureEMG.add_subplot(grid[rows - 1 - i])
                obj = self.model.EmgSensor[s].amplitude
                obj = obj.dropna()
                time = obj.index.to_numpy()
                amplitude = obj.values.flatten()
                ax.plot(time, amplitude, linewidth=0.6)

                # plot the title within the figure box
                xt = time[0]
                yt = (np.max(amplitude) - np.min(amplitude)) * 1.05
                yt += np.min(amplitude)
                ax.text(xt, yt, s.upper(), fontweight="bold")

                # set the x-axis limits and bounds
                time_rng = self._times[-1] - self._times[0]
                x_off = time_rng * 0.05
                ax.set_xlim(self._times[0] - x_off, self._times[-1] + x_off)
                ax.spines["bottom"].set_bounds(np.min(time), np.max(time))

                # set the y-axis limits
                amplitude_range = np.max(amplitude) - np.min(amplitude)
                y_off = amplitude_range * 0.05
                y_min = np.min(amplitude)
                y_max = np.max(amplitude)
                ax.set_ylim(y_min - y_off, y_max + y_off)
                ax.spines["left"].set_bounds(y_min, y_max)

                # share the x axis
                if i > 0:
                    ax.get_shared_x_axes().join(emg_axes[0], ax)
                    ax.set_xticklabels([])

                # adjust the layout
                ax.spines["right"].set_visible(False)
                ax.spines["top"].set_visible(False)
                if i == 0:
                    ax.set_xlabel("TIME", weight="bold")
                else:
                    ax.spines["bottom"].set_visible(False)
                    ax.xaxis.set_ticks([])

                # update the emg axes
                emg_axes += [ax]

                # plot the vertical lines
                x_line = [self._times[0], self._times[0]]
                y_line = [np.min(amplitude), np.max(amplitude)]
                self._emg[s] = ax.plot(x_line, y_line, "--", linewidth=0.5)[0]

            # prepare the emg vertical lines at each frame
            t = Thread(target=self._read_emg)
            t.start()

        else:

            # generate an empty widget
            self.canvasEMG = qtw.QWidget()

        # create the 3D model pane
        if self.model.has_ForcePlatform3D() or self.model.has_Marker3D():

            # generate the axis
            self._figure3D = pl.figure(dpi=self._dpi)
            self._figure3D.tight_layout()
            axis3D = self._figure3D.add_subplot(projection="3d")

            # set the layout
            axis3D.view_init(elev=10, azim=45)
            axis3D.set_xlabel("X", weight="bold")
            axis3D.set_ylabel("Y", weight="bold")
            axis3D.set_zlabel("Z", weight="bold")

            # set the axis limits
            if self.model.has_Marker3D():
                edges = self._data.loc[self._data.isin({"Sensor": ["Marker3D"]}).any(1)]
            else:
                idx = {"Sensor": ["ForcePlatform3D"], "Source": ["origin"]}
                edges = self._data.loc[
                    self._data[["Sensor", "Source"]].isin(idx).all(1)
                ]
            edges = edges["Amplitude"].values.flatten()
            maxc = max(0, np.nanmax(edges))
            minc = min(0, np.nanmin(edges))
            axis3D.set_xlim(minc, maxc)
            axis3D.set_ylim(minc, maxc)
            axis3D.set_zlim(minc, maxc)

            # force amplitude scaler
            if self.model.has_ForcePlatform3D():
                max_range = maxc - minc
                idx = {"Sensor": ["ForcePlatform3D"], "Source": ["amplitude"]}
                amp = self._data.loc[self._data[["Sensor", "Source"]].isin(idx).all(1)]
                amplitude = np.nanmax(amp["Amplitude"].values.flatten())
                force3D_sclr = max_range / amplitude
            else:
                force3D_sclr = 1

            # set the text objects
            df1 = self._data.isin({"Sensor": ["Marker3D", "ForcePlatform3D"]}).any(1)
            df1 = self._data.loc[df1]
            self._text3D = {}
            for label in np.unique(df1["Label"].values.flatten()):
                ax = axis3D.text(0, 0, 0, label, alpha=0.5)
                self._text3D[label] = ax

            # set the markers objects
            self._marker3D = {}
            df1 = self._data.loc[self._data.isin({"Sensor": ["Marker3D"]}).any(1)]
            for label in np.unique(df1["Label"].values.flatten()):
                ax = axis3D.plot(
                    0,
                    0,
                    0,
                    marker="o",
                    alpha=1.0,
                    color="navy",
                )
                self._marker3D[label] = ax[0]
            t_marker = Thread(target=self._read_marker)
            t_marker.start()

            # set the link objects
            self._link3D = {}
            df1 = self._data.loc[self._data.isin({"Sensor": ["Link3D"]}).any(1)]
            for label in np.unique(df1["Label"].values.flatten()):
                ax = axis3D.plot(
                    [0, 0],
                    [0, 0],
                    [0, 0],
                    alpha=0.7,
                    color="darkred",
                )
                self._link3D[label] = ax[0]
            t_link = Thread(target=self._read_link)
            t_link.start()

            # set the force platform objects
            self._force3D = {}
            df1 = self._data.loc[
                self._data.isin({"Sensor": ["ForcePlatform3D"]}).any(1)
            ]
            for label in np.unique(df1["Label"].values.flatten()):
                self._force3D[label] = axis3D.quiver(
                    [0],
                    [0],
                    [0],
                    [0],
                    [0],
                    [0],
                    color="darkgreen",
                    alpha=0.7,
                )

        # image pane
        splitter = qtw.QSplitter(qtc.Qt.Vertical)
        self.model3D = FigureCanvasQTAgg(self._model3D)
        self.canvasEMG = FigureCanvasQTAgg(self._figureEMG)
        splitter.addWidget(self.model3D)
        splitter.addWidget(self.canvasEMG)

        # widget layout
        layout = qtw.QVBoxLayout()
        layout.addWidget(splitter)
        layout.addWidget(commands_widget)
        self.setLayout(layout)
        self.setGraphicsEffect(opacity_zero)

        # update the figure
        self._update_data()

        # figure updaters
        if self.model.has_Marker3D():
            t_markers = qtc.QTimer()
            t_markers.timeout.connect(self._update_markers)
            self._threads += [t_markers]
        if self.model.has_ForcePlatform3D():
            t_forces = qtc.QTimer()
            t_forces.timeout.connect(self._update_forces)
            self._threads += [t_forces]
        if self.model.has_Link3D():
            t_links = qtc.QTimer()
            t_links.timeout.connect(self._update_links)
            self._threads += [t_links]
        if self.model.has_EmgSensor():
            t_emgs = qtc.QTimer()
            t_emgs.timeout.connect(self._update_emgs)
            self._threads += [t_emgs]
        t_timer = qtc.QTimer()
        t_timer.timeout.connect(self._update_timer)
        self._threads += [t_timer]
        self._play_timer = qtc.QTimer()
        self._play_timer.timeout.connect(self._move_forward)
        for t in self._threads:
            t.start()
        self._update_figure()

    def _get_source(self, df):
        """
        extract the values from a specific source.

        Parameters
        ----------
        data: DataFrame
            the data from which the info have to be extracted.

        Returns
        -------
        x, y, z, labels: array-like
            4 arrays containing the data of the source.
        """
        dd = df.pivot("Label", "Dimension", "Amplitude")
        xs, ys, zs = dd.values.T
        ts = dd.index.to_numpy()
        return xs, ys, zs, ts

    def _read_emg(self):
        """
        function used to get the positions to be updated on the rendered
        time frame.
        """
        for time in self._times:
            x_data = [time, time]
            vals = {i: x_data for i in self._emg}
            self.frames[time]["EmgSensor"] = vals

    def _read_marker(self):
        """
        function used to get the positions to be updated on the rendered
        time frame.
        """
        df0 = self._data.groupby(["Time", "Sensor", "Source"])
        for time in self._times:
            time = self._times[self._actual_frame]
            df = df0.get_group((time, "Marker3D", "coordinates"))
            xs, ys, zs, ts = self._get_source(df)
            self.frames[time]["Text3D"] = {}
            self.frames[time]["Marker3D"] = {}
            for x, y, z, t in zip(xs, ys, zs, ts):
                if np.any(np.isnan([x, y, z])):
                    self.frames[time]["Text3D"][t] = None
                    self.frames[time]["Marker3D"][t] = None
                else:
                    self.frames[time]["Text3D"][t] = (x, y, z)
                    self.frames[time]["Marker3D"][t] = (x, y, z)

    def _read_link(self):
        """
        function used to get the positions to be updated on the rendered
        time frame.
        """
        df = self._data.groupby(["Time", "Sensor", "Source"])
        for time in self._times:
            df0 = df.get_group((time, "Link3D", "p0"))
            df1 = df.get_group((time, "Link3D", "p1"))
            x0, y0, z0, ts = self._get_source(df0)
            x1, y1, z1, _ = self._get_source(df1)
            xs = np.vstack([x0, x1]).T
            ys = np.vstack([y0, y1]).T
            zs = np.vstack([z0, z1]).T
            self.frames[time]["Link3D"] = {}
            for x, y, z, t in zip(xs, ys, zs, ts):
                if np.any(np.isnan(np.concatenate([x, y, z]))):
                    self.frames[time]["Link3D"][t] = None
                else:
                    self.frames[time]["Link3D"][t] = (x, y, z)

    def is_running(self):
        """
        check if the player is running.
        """
        return self._is_running

    def _render_button(self, icon, event_handler=None):
        """
        private method used to generate valid buttons for this widget.

        Parameters
        ----------
        icon: str
            the path to the image used for the button

        event_handler: function
            the method to be passed as event handler for having the
            button being pressed.

        Returns
        -------
        obj: qtw.QPushButton
            a novel PushButton object.
        """
        button = qtw.QPushButton("")
        button.setFixedHeight = self._button_size
        button.setFixedWidth = self._button_size
        button.setFont(qtg.QFont("Arial", 0))
        pixmap = qtg.QPixmap(icon)
        pixmap = pixmap.scaled(self._button_size, self._button_size)
        qicon = qtg.QIcon(pixmap)
        button.setIcon(qicon)
        if event_handler is not None:
            button.clicked.connect(event_handler)
        return button

    def _move_forward(self):
        """
        function handling the press of the play button.
        """
        frame = self.slider.value()
        next_frame = frame + 1
        if next_frame > self.slider.maximum():
            if self.repeat_button.isChecked():
                next_frame = 0
            else:
                next_frame = frame
        self.slider.setValue(next_frame)

    def _move_backward(self):
        """
        function handling the press of the play button.
        """
        frame = self.slider.value()
        next_frame = frame - 1
        if next_frame < 0:
            if self.repeat_button.isChecked():
                next_frame = self.slider.maximum()
            else:
                next_frame = frame
        self.slider.setValue(next_frame)

    def _start_player(self):
        """
        stop the player
        """
        self._play_timer.start(self._update_rate)
        self._is_running = True
        icon_path = os.path.sep.join([self._path, "icons", "pause.png"])
        pxmap = qtg.QPixmap(icon_path)
        pxmap = pxmap.scaled(self._button_size, self._button_size)
        icon = qtg.QIcon(pxmap)
        self.play_button.setIcon(icon)

    def _stop_player(self):
        """
        stop the player
        """
        self._play_timer.stop()
        self._is_running = False
        icon_path = os.path.sep.join([self._path, "icons", "play.png"])
        pxmap = qtg.QPixmap(icon_path)
        pxmap = pxmap.scaled(self._button_size, self._button_size)
        icon = qtg.QIcon(pxmap)
        self.play_button.setIcon(icon)

    def _play_pressed(self):
        """
        method handling the play button press events.
        """
        if self.is_running():
            self._stop_player()
        else:
            self._start_player()

    def _forward_pressed(self):
        """
        method handling the forward button press events.
        """
        self._stop_player()
        self._move_forward()

    def _backward_pressed(self):
        """
        method handling the forward button press events.
        """
        self._stop_player()
        self._move_backward()

    def _slider_moved(self):
        """
        event handler for the slider value update.
        """
        self._actual_frame = self.slider.value()
        self._update_figure()

    def _update_markers(self):
        """
        function used to update the Marker3D data.
        """
        time = self._times[self._actual_frame]
        df = self._data.get_group((time, "Marker3D", "coordinates"))
        xs, ys, zs, ts = self._get_source(df)
        for x, y, z, t in zip(xs, ys, zs, ts):
            if np.any(np.isnan([x, y, z])):
                self._text3D[t]._visible = False
                self._marker3D[t]._visible = False
            else:
                self._text3D[t]._visible = True
                self._text3D[t]._x = x
                self._text3D[t]._y = y
                self._text3D[t]._z = z
                self._marker3D[t]._visible = True
                self._marker3D[t].set_data_3d(x, y, z)

    def _update_links(self):
        """
        function used to update the Link3D data.
        """
        time = self._times[self._actual_frame]
        df0 = self._data.get_group((time, "Link3D", "p0"))
        df1 = self._data.get_group((time, "Link3D", "p1"))
        x0, y0, z0, ts = self._get_source(df0)
        x1, y1, z1, _ = self._get_source(df1)
        xs = np.vstack([x0, x1]).T
        ys = np.vstack([y0, y1]).T
        zs = np.vstack([z0, z1]).T
        for x, y, z, t in zip(xs, ys, zs, ts):
            if np.any(np.isnan(np.concatenate([x, y, z]))):
                self._link3D[t]._visible = False
            else:
                self._link3D[t]._visible = True
                self._link3D[t].set_data_3d(x, y, z)

    def _update_forces(self):
        """
        function used to update the ForcePlatform3D data.
        """
        time = self._times[self._actual_frame]
        ori = self._data.get_group((time, "ForcePlatform3D", "origin"))
        amp = self._data.get_group((time, "ForcePlatform3D", "origin"))
        xs, ys, zs, ts = self._get_source(ori)
        us, vs, ws, _ = self._get_source(amp * self._force3D_sclr)
        for x, y, z, u, v, w, t in zip(xs, ys, zs, us, vs, ws, ts):
            if np.any(np.isnan(np.concatenate([x, y, z, u, v, w]))):
                self._force3D[t]._visible = False
                self._text3D[t]._visible = False
            else:
                self._force3D[t]._visible = True
                self._force3D[t].set_data_3d(x, y, z, u, v, z)
                self._text3D[t]._visible = True
                self._text3D[t]._x = x
                self._text3D[t]._y = y
                self._text3D[t]._z = z

    def _update_emgs(self):
        """
        function used to update the emg data.
        """
        time = self._times[self._actual_frame]
        for i in self._emg:
            self._emg[i].set_xdata([time, time])

    def _update_timer(self):
        """
        function used to update the timer.
        """
        time = self._times[self._actual_frame]
        minutes = time // 60000
        seconds = (time - minutes * 60000) // 1000
        msec = time - minutes * 60000 - seconds * 1000
        lbl = "{:02d}:{:02d}.{:03d}".format(minutes, seconds, msec)
        self.time_label.setText(lbl)

    def _update_data(self):
        """
        private function used to update the data when new sensors are
        appended.
        """

        # make the figure
        rows = len(self.model.EmgSensor) if self.model.has_EmgSensor() else 1
        if self.model.has_Marker3D() or self.model.has_ForcePlatform3D():
            cols = 3
        else:
            cols = 1
        grid = GridSpec(rows, cols)
        self._model3D = pl.figure(dpi=self._dpi)
        self._model3D.tight_layout(rect=(0.025, 0.025, 1, 0.975))

        # generate the 3D axis
        df0 = self.model.stack()
        if self.model.has_ForcePlatform3D() or self.model.has_Marker3D():

            # generate the axis
            self._axis3D = self._model3D.add_subplot(
                grid[:, :2],
                projection="3d",
            )

            # set the camera view
            self._axis3D.view_init(elev=10, azim=45)

            # set the axis label
            self._axis3D.set_xlabel("X", weight="bold")
            self._axis3D.set_ylabel("Y", weight="bold")
            self._axis3D.set_zlabel("Z", weight="bold")

            # set the axis limits
            if self.model.has_Marker3D():
                edges = df0.loc[df0.isin({"Sensor": ["Marker3D"]}).any(1)]
            else:
                idx = {"Sensor": ["ForcePlatform3D"], "Source": ["origin"]}
                edges = df0.loc[df0[["Sensor", "Source"]].isin(idx).all(1)]
            edges = edges["Amplitude"].values.flatten()
            maxc = max(0, np.nanmax(edges))
            minc = min(0, np.nanmin(edges))
            self._axis3D.set_xlim(minc, maxc)
            self._axis3D.set_ylim(minc, maxc)
            self._axis3D.set_zlim(minc, maxc)

            # force amplitude scaler
            if self.model.has_ForcePlatform3D():
                max_range = maxc - minc
                idx = {"Sensor": ["ForcePlatform3D"], "Source": ["amplitude"]}
                amp = df0.loc[df0[["Sensor", "Source"]].isin(idx).all(1)]
                amplitude = np.nanmax(amp["Amplitude"].values.flatten())
                self._force3D_sclr = max_range / amplitude

        # set the text objects
        df1 = df0.isin({"Sensor": ["Marker3D", "ForcePlatform3D"]}).any(1)
        df1 = df0.loc[df1]
        self._text3D = {}
        for label in np.unique(df1["Label"].values.flatten()):
            ax = self._axis3D.text(0, 0, 0, label, alpha=0.5)
            self._text3D[label] = ax

        # set the link objects
        self._link3D = {}
        df1 = df0.loc[df0.isin({"Sensor": ["Link3D"]}).any(1)]
        for label in np.unique(df1["Label"].values.flatten()):
            ax = self._axis3D.plot(
                [0, 0],
                [0, 0],
                [0, 0],
                alpha=0.7,
                color="darkred",
            )
            self._link3D[label] = ax[0]

        # set the markers objects
        self._marker3D = {}
        df1 = df0.loc[df0.isin({"Sensor": ["Marker3D"]}).any(1)]
        for label in np.unique(df1["Label"].values.flatten()):
            ax = self._axis3D.plot(
                0,
                0,
                0,
                marker="o",
                alpha=1.0,
                color="navy",
            )
            self._marker3D[label] = ax[0]

        # set the force platform objects
        self._force3D = {}
        df1 = df0.loc[df0.isin({"Sensor": ["ForcePlatform3D"]}).any(1)]
        for label in np.unique(df1["Label"].values.flatten()):
            self._force3D[label] = self._axis3D.quiver(
                [0],
                [0],
                [0],
                [0],
                [0],
                [0],
                color="darkgreen",
                alpha=0.7,
            )

        # get the time and frames
        self._data = df0.groupby(["Time", "Sensor", "Source"])
        idx = {"Sensor": ["ForcePlatform3D", "Marker3D", "Link3D"]}
        times = df0.loc[df0.isin(idx).any(1)]["Time"].values.flatten()
        self._times = np.unique(times)
        self._frames = np.arange(len(self._times))
        self._actual_frame = self._frames[0]

        # add the emg data
        self._emg = {}
        emg_axes = []
        if self.model.has_EmgSensor():
            n = len(self.model.EmgSensor)
            for i, s in enumerate(self.model.EmgSensor):

                # plot the whole EMG signal
                ax = self._model3D.add_subplot(grid[n - 1 - i, cols - 1])
                obj = self.model.EmgSensor[s].amplitude
                obj = obj.dropna()
                time = obj.index.to_numpy()
                amplitude = obj.values.flatten()
                ax.plot(time, amplitude, linewidth=0.6)

                # plot the title within the figure box
                xt = time[0]
                yt = (np.max(amplitude) - np.min(amplitude)) * 1.05
                yt += np.min(amplitude)
                ax.text(xt, yt, s.upper(), fontweight="bold")

                # set the x-axis limits and bounds
                time_rng = times[-1] - times[0]
                x_off = time_rng * 0.05
                ax.set_xlim(times[0] - x_off, times[-1] + x_off)
                ax.spines["bottom"].set_bounds(np.min(time), np.max(time))

                # set the y-axis limits
                amplitude_range = np.max(amplitude) - np.min(amplitude)
                y_off = amplitude_range * 0.05
                y_min = np.min(amplitude)
                y_max = np.max(amplitude)
                ax.set_ylim(y_min - y_off, y_max + y_off)
                ax.spines["left"].set_bounds(y_min, y_max)

                # share the x axis
                if i > 0:
                    ax.get_shared_x_axes().join(emg_axes[0], ax)
                    ax.set_xticklabels([])

                # adjust the layout
                ax.spines["right"].set_visible(False)
                ax.spines["top"].set_visible(False)
                if i == 0:
                    ax.set_xlabel("TIME", weight="bold")
                else:
                    ax.spines["bottom"].set_visible(False)
                    ax.xaxis.set_ticks([])

                # plot the vertical lines
                x_line = [times[0], times[0]]
                y_line = [np.min(amplitude), np.max(amplitude)]
                self._emg[s] = ax.plot(x_line, y_line, "--", linewidth=0.5)[0]

                # update the emg axes
                emg_axes += [ax]

        # update the slider
        self.slider.setMaximum(len(self._frames) - 1)

    def _update_figure(self):
        """
        update the actual rendered figure.
        """
        self._model3D.canvas.draw()
        self._model3D.canvas.flush_events()
