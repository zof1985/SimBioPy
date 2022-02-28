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
    canvas = None
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
        self._play_timer.timeout.connect(self._move_forward)

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
        opacity_zero = qtw.QGraphicsOpacityEffect()
        opacity_zero.setOpacity(0)
        commands_widget.setGraphicsEffect(opacity_zero)

        # update the figure
        self._update_data()

        # image pane
        self.canvas = FigureCanvasQTAgg(self._figure)

        # widget layout
        layout = qtw.QVBoxLayout()
        layout.addWidget(self.canvas)
        layout.addWidget(commands_widget)
        self.setLayout(layout)
        self.setGraphicsEffect(opacity_zero)

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

    def _update_slider(self):
        """
        event handler for the slider value update.
        """
        self._actual_frame = self.slider.value()
        self._update_figure()

    def _get_frame(self, index):
        """
        return the required frame.

        Parameters
        ----------
        index: int
            the index corresponding to the wanted frame.

        Returns
        -------
        frame: dict
            the data resulting from the required frame.
        """
        frame = self._frames[index]
        calculated = all([i is not None for i in frame.values()])
        if not calculated:

            def get_source(data, src):
                dd = data.get_group(src)
                dd = dd.pivot("Label", "Dimension", "Amplitude")
                xs, ys, zs = dd.values.T
                ts = dd.index.to_numpy()
                return xs, ys, zs, ts

            # get the output frame
            df1 = self._data.get_group(frame["Time"]).groupby("Sensor")
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
                    frame[sensor] = {}
                    for x, y, z, t in zip(xs, ys, zs, ts):
                        if np.any(np.isnan([x, y, z])):
                            frame[sensor][t] = None
                        else:
                            frame[sensor][t] = (x, y, z)

                if sensor == "Marker3D":
                    xs, ys, zs, ts = get_source(df2, "coordinates")
                    ix = ~np.isnan(xs)
                    frame[sensor] = (xs[ix], ys[ix], zs[ix])
                    for i, x, y, z, t in zip(ix, xs, ys, zs, ts):
                        if i:
                            frame["Text3D"][t] = None
                        else:
                            frame["Text3D"][t] = (x, y, z)

                if sensor == "ForcePlatform3D":
                    xs, ys, zs, ts = get_source(df2, "origin")
                    us, vs, ws, _ = get_source(df2, "amplitude")
                    ix = ~np.isnan(xs)
                    x, y, z = np.meshgrid(xs[ix], ys[ix], zs[ix])
                    u, v, w = np.meshgrid(us[ix], vs[ix], ws[ix])
                    u = u * self._force_sclr
                    v = v * self._force_sclr
                    w = w * self._force_sclr
                    frame[sensor] = (x, y, z, u, v, w)
                    for x, y, z, u, v, w, t in zip(xs, ys, zs, us, vs, ws, ts):
                        if np.any(np.isnan([x, y, z])):
                            frame["Text3D"][t] = None
                        else:
                            j = (x + u) * 0.5
                            k = (y + v) * 0.5
                            l = (z + w) * 0.5
                            frame["Text3D"][t] = (j, k, l)

            self._frames[index] = frame

        return frame

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

            # set the camera view
            self._axis3D.view_init(elev=10, azim=45)

            # set the axis label
            self._axis3D.set_xlabel("X", weight="bold")
            self._axis3D.set_ylabel("Y", weight="bold")
            self._axis3D.set_zlabel("Z", weight="bold")

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

        # get the (empty) frames
        self._data = df.groupby(["Time"])
        times = list(self._data.groups.keys())
        links = df.groupby(["Sensor"])
        if any([i == "Link3D" for i in list(links.groups.keys())]):
            links = links.get_group("Link3D")["Label"].values.flatten()
            links = {i: None for i in np.unique(links)}
        else:
            links = {}
        frame = {
            "Marker3D": None,
            "ForcePlatform3D": None,
            "Link3D": links,
            "Text3D": {i: None for i in labels},
        }
        self._frames = []
        for i, t in enumerate(times):
            frm = frame.copy()
            frm["Time"] = t
            self._frames += [frm]
        self._actual_frame = 0

        # populate the EMG data
        self._axisEMG = []
        if self.model.has_EmgSensor():
            n = len(self.model.EmgSensor)
            for i, s in enumerate(self.model.EmgSensor):

                # plot the whole EMG signal
                ax = self._figure.add_subplot(grid[n - 1 - i, cols - 1])
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
                    ax.get_shared_x_axes().join(self._axisEMG[0], ax)
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
                vline = ax.plot(x_line, y_line, "--", linewidth=0.5)
                self._emg_vertical_lines += vline

                # store the axis data
                self._axisEMG += [ax]

        # populate the links and text data
        self._link3D = {i: None for i in links}
        self._text3D = {i: None for i in labels}

        # update the slider
        self.slider.setMaximum(len(self._frames) - 1)

        # update the actual figure
        self._figure.tight_layout(rect=(0.025, 0.025, 1, 0.975))
        self._update_figure()

    def _update_figure(self):
        """
        update the actual rendered figure.
        """

        # get the actual frame
        frame = self._get_frame(self._actual_frame)

        # update the EMG plots
        time = frame["Time"]
        for i in range(len(self._emg_vertical_lines)):
            self._emg_vertical_lines[i].set_xdata([time, time])

        # update the 3D model
        for track, values in frame.items():
            if track == "Marker3D":
                if self._marker3D is None:
                    if values is not None:
                        self._marker3D = self._axis3D.scatter(*values)
                        self._marker3D.set_alpha(1)
                else:
                    if values is not None:
                        self._marker3D._offsets3d = values
                        self._marker3D._visible = True
                    else:
                        self._marker3D._visible = False

            if track == "ForcePlatform3D":
                if self._force3D is None:
                    if values is not None:
                        self._force3D = self._axis3D.quiver(*values)
                        self._force3D.set_alpha(0.75)
                else:
                    if values is not None:
                        self._force3D._offsets3d = values
                        self._force3D._visible = True
                    else:
                        self._force3D._visible = False

            if track == "Link3D":
                for lbl, vals in values.items():
                    if self._link3D[lbl] is None:
                        if vals is not None:
                            x, y, z = vals
                            ax = self._axis3D.plot(x, y, z, color="darkred")[0]
                            self._link3D[lbl] = ax
                            self._link3D[lbl].set_alpha(0.75)
                    else:
                        if vals is not None:
                            self._link3D[lbl].set_data_3d(*vals)
                            self._link3D[lbl]._visible = True
                        else:
                            self._link3D[lbl]._visible = False

            if track == "Text3D" and values is not None:
                for lbl, vals in values.items():
                    if self._text3D[lbl] is None:
                        if vals is not None:
                            x, y, z = vals
                            self._text3D[lbl] = self._axis3D.text(x, y, z, lbl)
                            self._text3D[lbl].set_alpha(0.5)
                    else:
                        if vals is not None:
                            x, y, z = vals
                            self._text3D[lbl]._x = x
                            self._text3D[lbl]._y = y
                            self._text3D[lbl]._z = z
                            self._text3D[lbl]._visible = True
                        else:
                            self._text3D[lbl]._visible = False

        # update the timer
        minutes = time // 60000
        seconds = (time - minutes * 60000) // 1000
        msec = time - minutes * 60000 - seconds * 1000
        lbl = "{:02d}:{:02d}.{:03d}".format(minutes, seconds, msec)
        self.time_label.setText(lbl)

        # update the figure
        self._figure.canvas.draw()
