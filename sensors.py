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
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
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
    canvas = None
    slider = None
    time_label = None
    play_button = None
    forward_button = None
    backward_button = None
    repeat_button = None

    # private variables
    _font_size = 10
    _play_timer = None
    _update_rate = 1  # msec
    _is_running = False
    _figure = None
    _axis3D = None
    _axisEMG = []
    _marker3D = None
    _force3D = None
    _link3D = {}
    _text3D = {}
    _force3D_sclr = 1
    _emg_vertical_lines = []
    _dpi = 300
    _actual_frame = None
    _frames = None

    def __init__(self, model, parent=None):
        """
        constructor
        """
        super(Model3DWidget, self).__init__(parent=parent)

        # store the model
        txt = "model must be a Model3D instance."
        assert isinstance(model, Model3D), txt
        self.model = model

        # timer
        self._play_timer = qtc.QTimer()
        self._play_timer.timeout.connect(self._player)

        # path to the package folder
        self._path = os.path.sep.join([os.getcwd(), "simbiopy"])

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
        self.slider.setMinimum = 0
        self.slider.setTickInterval(1)
        self.slider.valueChanged.connect(self._update_slider)

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
        commands_widget.setFixedHeight(40)

        # update the figure
        self._update_data()

        # image pane
        self.canvas = FigureCanvasQTAgg(self._figure)

        # widget layout
        layout = qtw.QVBoxLayout()
        layout.addWidget(self.canvas)
        layout.addWidget(commands_widget)
        self.setLayout(layout)

    def append(self, **objs):
        """
        append named objects to this model.

        Parameters
        ----------
        objs: keyword-named sensor
            any object being instance of the Sensor class.
        """
        self.model.append(**objs)
        self._update_data()

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
        button.setFixedHeight = 35
        button.setFixedWidth = 35
        button.setFont(qtg.QFont("Arial", 0))
        qicon = qtg.QIcon(qtg.QPixmap(icon).scaled(35, 35))
        button.setIcon(qicon)
        if event_handler is not None:
            button.clicked.connect(event_handler)
        return button

    def _player(self):
        """
        function handling the press of the play button.
        """
        next_value = self.slider.value() + self._update_rate
        if next_value > self.slider.maximum():
            next_value = 0
        self.slider.setValue(next_value)
        self._update_slider()

    def _start_player(self):
        """
        stop the player
        """
        self._play_timer.start(self._update_rate)
        self._is_running = True
        icon_path = os.path.sep.join([self._path, "icons", "pause.png"])
        icon = qtg.QIcon(qtg.QPixmap(icon_path).scaled(35, 35))
        self.play_button.setIcon(icon)

    def _stop_player(self):
        """
        stop the player
        """
        self._play_timer.stop()
        self._is_running = False
        icon_path = os.path.sep.join([self._path, "icons", "play.png"])
        icon = qtg.QIcon(qtg.QPixmap(icon_path).scaled(35, 35))
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

        # stop any running event
        self._stop_player()

        # check the actual repeat button selection
        if self.slider.maximum() > self._actual_frame:
            self.slider.setValue(self._actual_frame + 1)
        elif self.repeat_button.isChecked():
            self.slider.setValue(0)
        self._update_slider()

    def _backward_pressed(self):
        """
        method handling the forward button press events.
        """

        # stop any running event
        self._stop_player()

        # check the actual repeat button selection
        if self.slider.value() > 0:
            self.slider.setValue(self._actual_frame - 1)
        elif self.repeat_button.isChecked():
            self.slider.setValue(self.slider.maximum())
        self._update_slider()

    def _update_slider(self):
        """
        event handler for the slider value update.
        """
        self._actual_frame = self.slider.value()
        self._update_figure()

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
        self._figure = pl.figure(dpi=self._dpi)

        # handle the 3D data
        df = self.model.stack().groupby("Sensor")
        ss = []
        sensors3D = ["Marker3D", "ForcePlatform3D", "Link3D"]
        for i in sensors3D:
            try:
                ss += [df.get_group(i)]
            except KeyError:
                pass
        df = pd.concat(ss, axis=0)
        df = df.dropna(axis=0, inplace=False, how="all")

        # generate the 3D axis
        labels = df.groupby(["Sensor"])
        groups = list(labels.groups.keys())
        groups = [i for i in groups if i in ["Marker3D", "ForcePlatform3D"]]
        labels = pd.concat([labels.get_group(i) for i in groups], axis=0)
        labels = np.unique(labels["Label"].values.flatten())
        if len(labels) > 0:

            # generate the axis
            self._axis3D = self._figure.add_subplot(
                grid[:, :2],
                projection="3d",
            )

            # set the axis limits
            if self.model.has_Marker3D():
                edges = df.groupby("Sensor").get_group("Marker3D")
            else:
                edges = df.groupby(["Sensor", "Source"])
                edges = edges.get_group(("ForcePlatform3D", "origin"))
            edges = edges["Amplitude"].values.flatten()
            maxc = max(0, np.nanmax(edges))
            minc = min(0, np.nanmin(edges))
            self._axis3D.set_xlim(minc, maxc)
            self._axis3D.set_ylim(minc, maxc)
            self._axis3D.set_zlim(minc, maxc)

            # force amplitude scaler
            if self.model.has_ForcePlatform3D():
                max_range = maxc - minc
                amp = df.groupby(["Sensor", "Source"])
                amp = amp.get_group(("ForcePlatform3D", "amplitude"))
                amplitude = np.nanmax(amp["Amplitude"].values.flatten())
                self._force3D_sclr = max_range / amplitude

        # get the frames

        def get_source(data, src):
            dd = data.get_group(src)
            dd = dd.pivot("Label", "Dimension", "Amplitude")
            xs, ys, zs = dd.values.T
            ts = dd.index.to_numpy()
            return xs, ys, zs, ts

        frame = {i: () for i in sensors3D}
        frame["Text3D"] = {i: () for i in labels}
        frame["Time"] = ()
        self._frames = []
        df0 = df.groupby(["Time"])
        for time in list(df0.groups.keys()):
            df1 = df0.get_group(time).groupby("Sensor")
            frm = frame.copy()
            frm["Time"] = time
            for sensor in list(df1.groups.keys()):
                df2 = df1.get_group(sensor).groupby("Source")
                if sensor == "Link3D":
                    x0, y0, z0, t0 = get_source(df2, "p0")
                    x1, y1, z1, t1 = get_source(df2, "p1")
                    i0 = [i in t1 for i in t0]
                    i1 = [i in t0 for i in t1]
                    xs = np.vstack([x0[i0], x1[i1]]).T
                    ys = np.vstack([y0[i0], y1[i1]]).T
                    zs = np.vstack([z0[i0], z1[i1]]).T
                    ts = t1[i1]
                    frm[sensor] = {}
                    for x, y, z, t in zip(xs, ys, zs, ts):
                        frm[sensor][t] = (x, y, z)

                if sensor == "Marker3D":
                    xs, ys, zs, ts = get_source(df2, "coordinates")
                    frm[sensor] = (xs, ys, zs)
                    for x, y, z, t in zip(xs, ys, zs, ts):
                        frm["Text3D"][t] = (x, y, z)

                if sensor == "ForcePlatform3D":
                    xs, ys, zs, ts = get_source(df2, "origin")
                    xs, ys, zs = np.meshgrid(xs, ys, zs)
                    us, vs, ws, _ = get_source(df2, "amplitude")
                    us, vs, ws = np.meshgrid(us, vs, ws)
                    us = us * self._force_sclr
                    vs = vs * self._force_sclr
                    ws = ws * self._force_sclr
                    frm[sensor] = (xs, ys, zs, us, vs, ws)
                    js = (xs + us) * 0.5
                    ks = (ys + vs) * 0.5
                    ls = (zs + ws) * 0.5
                    for j, k, l, t in zip(js, ks, ls, ts):
                        frm["Text3D"][t] = (j, k, l)
            self._frames += [frm]

        # populate the plot
        self._actual_frame = self._frames[0]["Time"]
        frame = self._frames[self._actual_frame]
        for track, values in frame.items():
            if track == "Marker3D" and len(values) > 0:
                self._marker3D = self._axis3D.scatter(*values)

            if track == "ForcePlatform3D" and len(values) > 0:
                self._force3D = self._axis3D.quiver(*values)

            if track == "Link3D" and len(values) > 0:
                for lbl, vals in values.items():
                    self._link3D[lbl] = self._axis3D.plot(*vals)[0]

            if track == "Text3D" and len(values) > 0:
                for lbl, vals in values.items():
                    x, y, z = vals
                    self._text3D[lbl] = self._axis3D.text(x, y, z, lbl)

        # populate the EMG data
        self._axisEMG = []
        if self.model.has_EmgSensor():
            for i, s in enumerate(self.model.EmgSensor):

                # plot the whole EMG signal
                ax = self._figure.add_subplot(grid[i, cols - 1])
                ax.set_title(s, fontsize=self._font_size)
                obj = self.model.EmgSensor[s].amplitude
                obj = obj.dropna()
                time = obj.index.to_numpy()
                amplitude = obj.values.flatten()
                ax.plot(time, amplitude, linewidth=0.6)

                # plot the vertical lines
                x_line = [frame["Time"], frame["Time"]]
                y_line = [np.min(amplitude), np.max(amplitude)]
                vline = ax.plot(x_line, y_line, "--", linewidth=0.5)

                # set the x-axis limits
                ax.set_xlim(self._frames[0]["Time"], self._frames[-1]["Time"])

                # share the x axis
                if i > 0:
                    ax.get_shared_x_axes().join(self._axisEMG[0], ax)
                    ax.set_xticklabels([])

                # store the axis and the vertical line traces
                self._axisEMG += [ax]
                self._emg_vertical_lines += vline

        # update the actual figure
        self._figure.tight_layout()
        self._update_figure()

    def _update_figure(self):
        """
        update the actual rendered figure.
        """

        # get the actual frame
        frame = self._frames[self._actual_frame]

        # update the plots
        for track, values in frame.items():
            if track == "Marker3D" and len(values) > 0:
                self._marker3D._offsets3d = values

            if track == "ForcePlatform3D" and len(values) > 0:
                self._force3D._offsets3d = values

            if track == "Link3D":
                for lbl, vals in values.items():
                    if len(vals) > 0:
                        self._link3D[lbl].set_data_3d(*values)

            if track == "Text3D":
                for lbl, vals in values.items():
                    if len(vals) > 0:
                        x, y, z = vals
                        self._text3D[lbl].set_position_3d(x, y, z)
                        self._text3D[lbl].set_alpha(0.5)
                    else:
                        self._text3D[lbl].set_alpha(0.0)

        # populate the EMG data
        time = frame["Time"]
        for i in range(len(self._emg_vertical_lines)):
            self._emg_vertical_lines[i].set_xdata([time, time])

        # update the timer
        minutes = time // 60000
        seconds = (time - minutes * 60000) // 1000
        msec = time - minutes * 60000 - seconds * 1000
        lbl = "{:02d}:{:02d}.{:03d}".format(minutes, seconds, msec)
        self.time_label.setText(lbl)

        # update the figure
        self._figure.canvas.draw()
