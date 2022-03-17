# MODELS MODULE


#! IMPORTS


import os
import sys
import warnings
import numpy as np
import pandas as pd
import itertools as it
import PySide2.QtWidgets as qtw
import PySide2.QtCore as qtc
import PySide2.QtGui as qtg
import matplotlib.pyplot as pl
from matplotlib.gridspec import GridSpec
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib import use as matplotlib_use
from mpl_toolkits import mplot3d
from .geometry import *
from .sensors import *
from .utils import *


#! MATPLOTLIB OPTIONS


matplotlib_use("Qt5Agg")
pl.rc("font", size=3)  # controls default text sizes
pl.rc("axes", titlesize=3)  # fontsize of the axes title
pl.rc("axes", labelsize=3)  # fontsize of the x and y labels
pl.rc("xtick", labelsize=3)  # fontsize of the x tick labels
pl.rc("ytick", labelsize=3)  # fontsize of the y tick labels
pl.rc("legend", fontsize=3)  # legend fontsize
pl.rc("figure", titlesize=3)  # fontsize of the figure title
pl.rc("figure", autolayout=True)


#! CLASSES


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
            obj.append(**getattr(self, attr))
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

    def plot(self, high_dpi: bool = True) -> None:
        """
        plot the actual model.

        Parameters
        ----------
        high_dpi: bool
            enable high dpi scaling.
        """
        # highdpi scaling
        if high_dpi:
            qtw.QApplication.setAttribute(qtc.Qt.AA_EnableHighDpiScaling, True)
            qtw.QApplication.setAttribute(qtc.Qt.AA_UseHighDpiPixmaps, True)

        # app generation
        app = qtw.QApplication(sys.argv)
        widget = Model3DWidget(self)
        widget.show()
        app.exec_()

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


class FigureAnimator:
    """
    Speed up the redraw of animated artists contained in a figure.

    Parameters
    ----------
    figure: matplotlib.pyplot.Figure
        a matplotlib figure.

    artists: Iterable[Artist]
        an iterable of artists being those elements that will be updated
        on top of figure.
    """

    def __init__(self, figure, artists):
        """
        constructor
        """
        self.figure = figure
        self._background = None

        # get the animated artists
        self._artists = []
        for art in artists:
            art.set_animated(True)
            self._artists.append(art)

        # grab the background on every draw
        self._cid = self.figure.canvas.mpl_connect("draw_event", self.on_draw)

    def on_draw(self, event):
        """
        Callback to register with 'draw_event'.
        """
        if event is not None:
            if event.canvas != self.figure.canvas:
                raise RuntimeError
        bbox = self.figure.canvas.figure.bbox
        self._background = self.figure.canvas.copy_from_bbox(bbox)
        self._draw_animated()

    def _draw_animated(self):
        """
        Draw all of the animated artists.
        """
        for a in self._artists:
            self.figure.canvas.figure.draw_artist(a)

    def update(self):
        """
        Update the screen with animated artists.
        """

        # update the background if required
        if self._background is None:
            self.on_draw(None)

        else:

            # restore the background
            self.figure.canvas.restore_region(self._background)

            # draw all of the animated artists
            self._draw_animated()

            # update the GUI state
            self.figure.canvas.blit(self.figure.canvas.figure.bbox)

        # let the GUI event loop process anything it has to do
        self.figure.canvas.flush_events()


class OptionGroup(qtw.QGroupBox):
    """
    make a line option.

    Parameters
    ----------
    label: str
        the name of the option.

    zorder: int
        the order of visualization of each object.

    min_value: float
        the minimum acceptable value.

    max_value: float
        the maximum acceptable value.

    step_value: float
        the step increments accepted.

    default_value: float
        the starting value.

    default_color: str or tuple
        the default color.
    """

    # class variables
    label = None
    sizeSlider = None
    sizeBox = None
    colorBox = None
    font_size = None
    object_size = None
    zorderBox = None
    _color = None

    # signals
    zorderChanged = qtc.Signal()
    colorChanged = qtc.Signal()
    valueChanged = qtc.Signal()

    def __init__(
        self,
        label="",
        min_value=1,
        max_value=100,
        step_value=1,
        default_value=1,
        default_color=(255, 0, 0, 255),
        default_zorder=0,
        font_size=12,
        object_size=35,
    ):
        """
        constructor
        """
        super().__init__()

        # sizes
        self.font_size = font_size
        self.object_size = object_size

        # zorder label
        zorder_label = qtw.QLabel("Z Order")
        object_font = qtg.QFont("Arial", max(1, self.font_size - 2))
        zorder_label.setFont(object_font)
        zorder_label.setFixedHeight(self.object_size)
        zorder_label.setAlignment(qtc.Qt.AlignCenter | qtc.Qt.AlignBottom)

        # zorder box
        self.zorderBox = self.sizeBox = qtw.QSpinBox()
        self.zorderBox.setFont(object_font)
        self.zorderBox.setFixedHeight(self.object_size)
        self.zorderBox.setFixedWidth(self.object_size * 2)
        self.zorderBox.setMinimum(0)
        self.zorderBox.setMaximum(10)
        self.zorderBox.setSingleStep(1)
        self.zorderBox.setValue(default_zorder)
        self.zorderBox.setStyleSheet("border: none;")

        # zorder pane
        zorder_layout = qtw.QVBoxLayout()
        zorder_layout.setSpacing(0)
        zorder_layout.setContentsMargins(0, 0, 0, 0)
        zorder_layout.addWidget(zorder_label)
        zorder_layout.addWidget(self.zorderBox)
        zorder_widget = qtw.QWidget()
        zorder_widget.setLayout(zorder_layout)

        # size label
        size_label = qtw.QLabel("Size")
        size_label.setFont(object_font)
        size_label.setFixedHeight(self.object_size)
        size_label.setAlignment(qtc.Qt.AlignCenter | qtc.Qt.AlignBottom)

        # size slider
        self.sizeSlider = qtw.QSlider(qtc.Qt.Horizontal)
        self.sizeSlider.setMinimum(min_value * 10)
        self.sizeSlider.setMaximum(max_value * 10)
        self.sizeSlider.setTickInterval(step_value * 10)
        self.sizeSlider.setValue(default_value * 10)
        self.sizeSlider.setFixedHeight(self.object_size)
        self.sizeSlider.setFixedWidth(self.object_size * 5)
        self.sizeSlider.setStyleSheet("border: none;")

        # spinbox
        self.sizeBox = qtw.QDoubleSpinBox()
        self.sizeBox.setDecimals(1)
        self.sizeBox.setFont(object_font)
        self.sizeBox.setFixedHeight(self.object_size)
        self.sizeBox.setFixedWidth(self.object_size * 2)
        self.sizeBox.setMinimum(min_value)
        self.sizeBox.setMaximum(max_value)
        self.sizeBox.setSingleStep(step_value)
        self.sizeBox.setValue(default_value)
        self.sizeBox.setStyleSheet("border: none;")

        # size pane
        size_layout1 = qtw.QHBoxLayout()
        size_layout1.setSpacing(0)
        size_layout1.setContentsMargins(0, 0, 0, 0)
        size_layout1.addWidget(self.sizeSlider)
        size_layout1.addWidget(self.sizeBox)
        size_widget1 = qtw.QWidget()
        size_widget1.setLayout(size_layout1)
        size_layout2 = qtw.QVBoxLayout()
        size_layout2.setSpacing(0)
        size_layout2.setContentsMargins(0, 0, 0, 0)
        size_layout2.addWidget(size_label)
        size_layout2.addWidget(size_widget1)
        size_widget2 = qtw.QWidget()
        size_widget2.setLayout(size_layout2)

        # color label
        color_label = qtw.QLabel("Color")
        color_label.setFont(object_font)
        color_label.setFixedHeight(self.object_size)
        color_label.setAlignment(qtc.Qt.AlignCenter | qtc.Qt.AlignBottom)

        # color box
        self.colorBox = qtw.QPushButton()
        self.colorBox.setFixedHeight(self.object_size)
        self.colorBox.setFixedWidth(self.object_size)
        self.colorBox.setAutoFillBackground(True)
        self.colorBox.setStyleSheet("border: 0px;")
        self.setColor(default_color)

        # color pane
        color_layout = qtw.QVBoxLayout()
        color_layout.setSpacing(0)
        color_layout.setContentsMargins(0, 0, 0, 0)
        color_layout.addWidget(color_label)
        color_layout.addWidget(self.colorBox)
        color_widget = qtw.QWidget()
        color_widget.setLayout(color_layout)

        # option pane
        layout = qtw.QHBoxLayout()
        layout.addWidget(color_widget)
        layout.addWidget(size_widget2)
        layout.addWidget(zorder_widget)
        layout.setSpacing(10)
        layout.setContentsMargins(2, 0, 2, 2)
        self.setLayout(layout)
        self.setTitle(label)

        # connections
        self.sizeBox.valueChanged.connect(self.adjustSizeSlider)
        self.sizeSlider.valueChanged.connect(self.adjustSizeBox)
        self.colorBox.clicked.connect(self.adjustColor)
        self.zorderBox.valueChanged.connect(self.adjustZOrderBox)

    def zorder(self):
        """
        return the actual stored zorder.
        """
        return self.zorderBox.value()

    def size(self):
        """
        return the actual stored size.
        """
        return self.sizeBox.value()

    def color(self):
        """
        return the actual color stored.
        """
        return self._color.getRgbF()

    def adjustSizeSlider(self):
        """
        adjust the slider value according to the spinbox value.
        """
        if self.sizeSlider.value() != self.sizeBox.value():
            self.sizeSlider.setValue(self.sizeBox.value() * 10)

    def adjustSizeBox(self):
        """
        adjust the spinbox value according to the slider value.
        """
        if self.sizeSlider.value() != self.sizeBox.value():
            self.sizeBox.setValue(self.sizeSlider.value() / 10)
            self.valueChanged.emit()

    def adjustZOrderBox(self):
        """
        handle changes in the zorderBox.
        """
        self.zorderChanged.emit()

    def adjustColor(self):
        """
        select and set the desired color.
        """
        try:
            color = qtw.QColorDialog.getColor(
                initial=self._color,
                options=qtw.QColorDialog.ShowAlphaChannel,
            )
            if color.isValid():
                self.setColor(color)
        except Exception:
            pass

    def setColor(self, rgba):
        """
        set the required rgba color.

        Parameters
        ----------
        rgba: tuple
            the 4 elements tuple defining a color.
        """

        # check the input
        txt = "'color' must be a QColor object or a tuple/list of 4 elements "
        txt += "each in the 0-1 range."
        assert isinstance(rgba, (tuple, list, qtg.QColor)), txt
        if not isinstance(rgba, qtg.QColor):
            assert len(rgba) == 4, txt
            assert all([0 <= i <= 1 for i in rgba]), txt
            if isinstance(rgba, list):
                rgba = tuple(rgba)
            rgba = qtg.QColor.fromRgbF(*rgba)

        # create the QColor object
        self._color = rgba
        values = tuple((np.array(self._color.getRgbF()) * 255).astype(int))
        txt = "background-color: rgba{};".format(values)
        self.colorBox.setStyleSheet(txt)
        self.colorChanged.emit()


class Model3DWidget(qtw.QWidget):
    """
    renderer for a 3D Model.

    Parameters
    ----------
    model: simbiopy.models.Model3D
        the model to be visualized

    vertical_axis: str
        the label of the dimension corresponding to the vertical axis.

    parent: PySide2.QtWidgets.QWidget
        the parent widget
    """

    # options pane
    ground_options = None
    marker_options = None
    force_options = None
    link_options = None
    text_options = None
    reference_options = None
    emg_signal_options = None
    emg_bar_options = None
    options_pane = None

    # command bar
    home_button = None
    media_action = None
    marker_button = None
    force_button = None
    link_button = None
    text_button = None
    reference_button = None
    backward_button = None
    play_button = None
    forward_button = None
    speed_box = None
    repeat_button = None
    time_label = None
    progress_slider = None
    option_button = None
    ground_button = None

    # default settings
    _default_view = {
        "elev": 10,
        "azim": 45,
        "vertical_axis": "y",
    }
    _default_ground = {
        "color": (0.1, 0.1, 0.1, 0.05),
        "linewidth": 0.01,
        "zorder": 3,
    }
    _default_marker = {
        "color": (1, 0, 0, 1),
        "markersize": 1.0,
        "zorder": 1,
    }
    _default_force = {
        "color": (0, 1, 0, 0.7),
        "linewidth": 0.5,
        "zorder": 2,
    }
    _default_link = {
        "color": (0, 0, 1, 0.7),
        "linewidth": 0.5,
        "zorder": 3,
    }
    _default_text = {
        "color": (0, 0, 0, 0.5),
        "size": 3,
        "zorder": 0,
    }
    _default_ref = {
        "color": (1, 0.5, 0, 1),
        "linewidth": 0.5,
        "zorder": 0,
    }
    _default_emg_signal = {
        "color": (0, 0.5, 1, 0.7),
        "linewidth": 0.1,
        "zorder": 1,
    }
    _default_emg_bar = {
        "color": (1, 0, 0.5, 1),
        "linewidth": 0.5,
        "zorder": 0,
    }

    # class variables
    data = None
    canvas3D = None
    canvasEMG = None

    # private variables
    _dpi = 300
    _times = None
    _font_size = 12
    _button_size = 35
    _play_timer = None
    _update_rate = 1  # msec
    _is_running = False
    _figure3D = None
    _axis3D = None
    _figureEMG = None
    _axisEMG = None
    _Marker3D = {}
    _ForcePlatform3D = {}
    _Link3D = {}
    _ReferenceFrame3D = {}
    _EmgSensor = {}
    _Ground3D = None
    _actual_frame = None
    _FigureAnimator3D = None
    _FigureAnimatorEMG = None
    _play_start_time = None
    _path = None
    _limits = None
    _arrow_angle = 15  # degrees
    _arrow_length = 0.1  # 10% of the quiver length

    def __init__(
        self,
        model,
        vertical_axis: str = "Y",
        parent: qtw.QWidget = None,
    ):
        """
        constructor
        """
        super(Model3DWidget, self).__init__(parent=parent)

        # check the model
        txt = "model must be a Model3D instance."
        assert isinstance(model, Model3D), txt

        # path to the package folder
        self._path = os.path.sep.join([os.getcwd(), "simbiopy"])

        # set the actual frame
        self._actual_frame = 0

        # get the data
        dfs = {}
        cols = ["X", "Y", "Z"]
        m = np.array([i == vertical_axis for i in cols])
        m = m.astype(int)
        ix = np.where(m == 1)[0]
        times = []

        # markers
        if model.has_Marker3D():
            dfs["Marker3D"] = {}
            max_coords = 1
            times = []
            for i, v in model.Marker3D.items():
                dfs["Marker3D"][i] = v.pivot()
                new_coords = np.nanmax(v.coordinates.values.T[ix])
                max_coords = max(max_coords, new_coords)
                times += [v.index]
        else:
            max_coords = 1

        # links
        if model.has_Link3D():
            dfs["Link3D"] = {i: v.pivot() for i, v in model.Link3D.items()}

        # forces
        if model.has_ForcePlatform3D():
            dfs["ForcePlatform3D"] = {}

            # get the force scaler
            max_force = 1
            for lbl, sns in model.ForcePlatform3D.items():
                max_force = max(max_force, np.max(sns.force.values.T[ix]))
            scaler = max_coords / max_force

            # update all the force data
            for lbl, sns in model.ForcePlatform3D.items():
                p0 = sns.origin
                p1 = sns.force * scaler + p0
                dfs["ForcePlatform3D"][lbl] = self._makeArrow(p0, p1, ix)

            # set the time
            if len(times) == 0:
                for out in dfs["ForcePlatform3D"].values():
                    times += [out.index]

        # EMG
        if model.has_EmgSensor():
            dfs["EmgSensor"] = {}
            for i, v in model.EmgSensor.items():
                dfs["EmgSensor"][i] = v.pivot()
                dfs["EmgSensor"][i].insert(0, "MIN", np.min(v.amplitude.values))
                dfs["EmgSensor"][i].insert(0, "MAX", np.max(v.amplitude.values))

            # set the time
            if len(times) == 0:
                for out in dfs["EmgSensor"].values():
                    times += [out.index]

        # include the alphas at each sample
        def set_alpha(obj):
            if isinstance(obj, dict):
                return {i: set_alpha(v) for i, v in obj.items()}
            alphas = (~np.all(np.isnan(obj.values), axis=1)).astype(int)
            obj.insert(obj.shape[1], "A", alphas)
            return obj

        dfs = set_alpha(dfs)

        # get the timing
        self._times = np.unique(np.concatenate(times))

        # store the data
        self.data = dfs

        # make the EMG pane
        if model.has_EmgSensor():

            # figure
            rows = len(self.data["EmgSensor"])
            grid = GridSpec(rows, 1)
            self._figureEMG = pl.figure(dpi=self._dpi)
            self.canvasEMG = FigureCanvasQTAgg(self._figureEMG)

            # resizing event handler
            self._figureEMG.canvas.mpl_connect(
                "resize_event",
                self._resize_event,
            )

            # add the emg data
            self._EmgSensor = {}
            axes = []
            for i, s in enumerate(model.EmgSensor):
                self._EmgSensor[s] = {}

                # plot the whole EMG signal
                ax = self._figureEMG.add_subplot(grid[rows - 1 - i])
                obj = model.EmgSensor[s].amplitude
                obj = obj.dropna()
                time = obj.index.to_numpy()
                amplitude = obj.values.flatten()
                line = ax.plot(time, amplitude, linewidth=0.5)
                self._EmgSensor[s]["Signal"] = line[0]

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
                ax.spines["bottom"].set_linewidth(0.5)

                # set the y-axis limits and bounds
                amplitude_range = np.max(amplitude) - np.min(amplitude)
                y_off = amplitude_range * 0.05
                y_min = np.min(amplitude)
                y_max = np.max(amplitude)
                ax.set_ylim(y_min - y_off, y_max + y_off)
                ax.spines["left"].set_bounds(y_min, y_max)
                ax.spines["left"].set_linewidth(0.5)

                # share the x axis
                if i > 0:
                    ax.get_shared_x_axes().join(axes[0], ax)
                    ax.set_xticklabels([])

                # adjust the layout
                ax.spines["right"].set_visible(False)
                ax.spines["top"].set_visible(False)
                if i == 0:
                    ax.set_xlabel("TIME", weight="bold")
                else:
                    ax.spines["bottom"].set_visible(False)
                    ax.xaxis.set_ticks([])

                # set the ticks params
                ax.tick_params(
                    direction="out",
                    length=1,
                    width=0.5,
                    colors="k",
                    pad=1,
                )

                # plot the vertical lines
                x_line = [self._times[0], self._times[0]]
                y_line = [np.min(amplitude), np.max(amplitude)]
                self._EmgSensor[s]["Bar"] = ax.plot(
                    x_line,
                    y_line,
                    "--",
                    linewidth=0.4,
                    animated=True,
                )[0]

                # store the axis
                axes += [ax]

            # setup the figure animator object
            artists = [i["Bar"] for i in self._EmgSensor.values()]
            artists += [i["Signal"] for i in self._EmgSensor.values()]
            self._FigureAnimatorEMG = FigureAnimator(
                figure=self._figureEMG,
                artists=artists,
            )

        else:

            # generate an empty widget
            self.canvasEMG = qtw.QWidget()

        # create the 3D model pane
        if model.has_ForcePlatform3D() or model.has_Marker3D():

            # check the vertical axis input
            if any([i == "Marker3D" for i in list(self.data.keys())]):
                dim = list(self.data["Marker3D"].values())[0]
            else:
                dim = list(dfs["ForcePlatform3D"].values())[0]
            dim = np.array([i[1] for i in dim.columns.to_numpy()])
            txt = "vertical_axis not found in model."
            assert vertical_axis.upper() in dim, txt
            self._default_view["vertical_axis"] = vertical_axis.lower()

            # generate the axis
            self._figure3D = pl.figure(dpi=self._dpi)
            self.canvas3D = FigureCanvasQTAgg(self._figure3D)
            self._axis3D = self._figure3D.add_subplot(
                projection="3d",
                proj_type="persp",  # 'ortho'
                adjustable="box",  # 'datalim'
                frame_on=False,
            )
            self._tight = True

            # set the view
            self._axis3D.view_init(**self._default_view)

            # resizing event handler
            self._figure3D.canvas.mpl_connect(
                "resize_event",
                self._resize_event,
            )

            # make the axis lines transparent
            self._axis3D.xaxis.line.set_color((1, 1, 1, 0))
            self._axis3D.yaxis.line.set_color((1, 1, 1, 0))
            self._axis3D.zaxis.line.set_color((1, 1, 1, 0))

            # remove the ticks
            self._axis3D.xaxis.set_ticks([])
            self._axis3D.yaxis.set_ticks([])
            self._axis3D.zaxis.set_ticks([])

            # make the axes transparent
            for l in ["x", "y", "z"]:
                ax = eval("self._axis3D.w_{}axis".format(l))
                ax.set_pane_color((1, 1, 1, 0))
                ax._axinfo["grid"]["color"] = (1, 1, 1, 0)

            # set the initial limits
            if model.has_Marker3D():
                edges = [v.values for v in self.data["Marker3D"].values()]
            else:
                edges = [v for v in self.data["ForcePlatform3D"].values()]
                edges = [v.values[:, :3] for v in edges]
            edges = np.concatenate(edges, axis=0)
            maxc = max(0, np.nanmax(edges) * 1.5)
            minc = min(0, np.nanmin(edges) * 1.5)
            self._limits = (minc, maxc)
            self._axis3D.set_xlim(*self._limits)
            self._axis3D.set_ylim(*self._limits)
            self._axis3D.set_zlim(*self._limits)

            # plot the reference frame
            ref_scale = abs(np.diff(self._limits))[0] * 0.1
            r0 = pd.DataFrame([[0, 0, 0]], columns=cols)
            rx = pd.DataFrame([[ref_scale, 0, 0]], columns=cols)
            ry = pd.DataFrame([[0, ref_scale, 0]], columns=cols)
            rz = pd.DataFrame([[0, 0, ref_scale]], columns=cols)
            i = self._makeArrow(r0, rx, ix)
            j = self._makeArrow(r0, ry, ix)
            k = self._makeArrow(r0, rz, ix)
            self._ReferenceFrame3D = {}
            for lbl, val in zip(cols, [i, j, k]):
                self._ReferenceFrame3D[lbl] = self._plotArrow(
                    val,
                    self._axis3D,
                    **self._default_ref,
                )
                u, v, w = val.values.flatten()[3:6] * 1.4
                self._ReferenceFrame3D[lbl]["Label"] = self._axis3D.text(
                    u,
                    v,
                    w,
                    lbl,
                    animated=True,
                    ha="center",
                    va="center",
                    **self._default_text,
                )
                x, y, z = val.values.flatten()[:3]
                self._ReferenceFrame3D[lbl]["Point"] = self._axis3D.plot(
                    x,
                    y,
                    z,
                    marker="o",
                    animated=True,
                    **self._default_marker,
                )[0]

            # plot forces
            if model.has_ForcePlatform3D():
                self._ForcePlatform3D = {}
                for lbl, val in self.data["ForcePlatform3D"].items():
                    df = val.loc[self._times[self._actual_frame]]
                    self._ForcePlatform3D[lbl] = self._plotArrow(
                        df=df,
                        ax=self._axis3D,
                        **self._default_force,
                    )
                    x, y, z = df.values.flatten()[:3]
                    self._ForcePlatform3D[lbl]["Point"] = self._axis3D.plot(
                        x,
                        y,
                        z,
                        marker="o",
                        animated=True,
                        **self._default_marker,
                    )[0]
                    self._ForcePlatform3D[lbl]["Label"] = self._axis3D.text(
                        x,
                        y,
                        z,
                        lbl,
                        animated=True,
                        **self._default_text,
                    )

            # plot markers
            if model.has_Marker3D():
                self._Marker3D = {}
                for lbl, val in self.data["Marker3D"].items():
                    df = val.loc[self._times[self._actual_frame]]
                    x, y, z, _ = df.values.flatten()
                    self._Marker3D[lbl] = {
                        "Point": self._axis3D.plot(
                            x,
                            y,
                            z,
                            marker="o",
                            animated=True,
                            **self._default_marker,
                        )[0],
                        "Label": self._axis3D.text(
                            x,
                            y,
                            z,
                            lbl,
                            animated=True,
                            **self._default_text,
                        ),
                    }

            # plot links
            if model.has_Link3D():
                self._Link3D = {}
                for lbl, val in self.data["Link3D"].items():
                    df = val.loc[self._times[self._actual_frame]]
                    x0, y0, z0, x1, y1, z1, _ = df.values.flatten()
                    self._Link3D[lbl] = {
                        "Line": self._axis3D.plot(
                            np.array([x0, x1]),
                            np.array([y0, y1]),
                            np.array([z0, z1]),
                            animated=True,
                            **self._default_link,
                        )[0],
                    }

            # plot the ground
            if model.has_Marker3D():
                dt = self.data["Marker3D"].values()
                dt = np.concatenate([v.values for v in dt], axis=0).T[:3]
                mx, my, mz = dt
            else:
                mx = np.array([])
                my = np.array([])
                mz = np.array([])
            if model.has_ForcePlatform3D():
                dt = self.data["ForcePlatform3D"].values()
                dt = np.concatenate([v.values for v in dt], axis=0).T[:3]
                fx, fy, fz = dt
            else:
                fx = np.array([])
                fy = np.array([])
                fz = np.array([])
            x = np.concatenate([mx, fx], axis=0)
            y = np.concatenate([my, fy], axis=0)
            z = np.concatenate([mz, fz], axis=0)
            frame = {
                "x": np.linspace(np.nanmin(x), np.nanmax(x), 20),
                "y": np.linspace(np.nanmin(y), np.nanmax(y), 20),
                "z": np.linspace(np.nanmin(z), np.nanmax(z), 20),
            }
            if self._default_view["vertical_axis"] == "x":
                sY, sZ = np.meshgrid(frame["y"], frame["z"])
                sZ = np.zeros_like(sY)
            elif self._default_view["vertical_axis"] == "y":
                sX, sZ = np.meshgrid(frame["x"], frame["z"])
                sY = np.zeros_like(sX)
            else:
                sX, sY = np.meshgrid(frame["x"], frame["y"])
                sZ = np.zeros_like(sX)
            self._Ground3D = {
                "Ground": {
                    "Grid": self._axis3D.plot_wireframe(
                        sX,
                        sY,
                        sZ,
                        animated=True,
                    ),
                },
            }

            # setup the Figure Animator
            artists = []
            objs = [self._Marker3D, self._ForcePlatform3D, self._Link3D]
            objs += [self._ReferenceFrame3D, self._Ground3D]
            for elem in objs:
                for objs in elem.values():
                    for ax in objs.values():
                        artists += [ax]
            self._FigureAnimator3D = FigureAnimator(
                figure=self._figure3D,
                artists=artists,
            )

        else:

            # generate an empty widget
            self.canvas3D = qtw.QWidget()

        # wrap the two figures into a splitted view
        splitter = qtw.QSplitter(qtc.Qt.Horizontal)
        splitter.addWidget(self.canvas3D)
        splitter.addWidget(self.canvasEMG)

        # create a commands bar
        commands_bar = qtw.QToolBar()
        commands_bar.setStyleSheet("spacing: 10px;")

        # add the home function
        self.home_button = self._command_button(
            tip="Reset the view to default.",
            icon=os.path.sep.join([self._path, "icons", "home.png"]),
            enabled=True,
            checkable=False,
            fun=self._home_pressed,
        )
        commands_bar.addWidget(self.home_button)

        # multimedia action button
        media_icon = os.path.sep.join([self._path, "icons", "media.png"])
        media_icon = self._makeIcon(media_icon)
        self.media_action = qtw.QAction(media_icon, "", None)
        self.media_action.triggered.connect(self._media_action_pressed)
        self.media_action.setToolTip("Exporting options.")
        self.media_action.setEnabled(False)  #! DEBUG
        commands_bar.addAction(self.media_action)

        # add a separator
        commands_bar.addSeparator()

        # function show/hide the ground
        self.ground_button = self._command_button(
            tip="Show/Hide the ground.",
            icon=os.path.sep.join([self._path, "icons", "ground.png"]),
            enabled=model.has_Marker3D() or model.has_ForcePlatform3D(),
            checkable=True,
            fun=self._ground_button_pressed,
        )
        commands_bar.addWidget(self.ground_button)

        # function show/hide markers
        self.marker_button = self._command_button(
            tip="Show/Hide the Marker3D objects.",
            icon=os.path.sep.join([self._path, "icons", "markers.png"]),
            enabled=model.has_Marker3D(),
            checkable=True,
            fun=self._marker_button_pressed,
        )
        commands_bar.addWidget(self.marker_button)

        # function show/hide forces function
        self.force_button = self._command_button(
            tip="Show/Hide the ForcePlatform3D objects.",
            icon=os.path.sep.join([self._path, "icons", "forces.png"]),
            enabled=model.has_ForcePlatform3D(),
            checkable=True,
            fun=self._force_button_pressed,
        )
        commands_bar.addWidget(self.force_button)

        # function show/hide links function
        self.link_button = self._command_button(
            tip="Show/Hide the Link3D objects.",
            icon=os.path.sep.join([self._path, "icons", "links.png"]),
            enabled=model.has_Link3D(),
            checkable=True,
            fun=self._link_button_pressed,
        )
        commands_bar.addWidget(self.link_button)

        # function show/hide labels function
        self.text_button = self._command_button(
            tip="Show/Hide the labels.",
            icon=os.path.sep.join([self._path, "icons", "txt.png"]),
            enabled=model.has_ForcePlatform3D() | model.has_Marker3D(),
            checkable=True,
            fun=self._text_button_pressed,
        )
        commands_bar.addWidget(self.text_button)

        # function show/hide reference function
        self.reference_button = self._command_button(
            tip="Show/Hide the reference frame.",
            icon=os.path.sep.join([self._path, "icons", "reference.png"]),
            enabled=True,
            checkable=True,
            fun=self._reference_pressed,
        )
        commands_bar.addWidget(self.reference_button)

        # add a separator
        commands_bar.addSeparator()

        # add the move backward function
        self.backward_button = self._command_button(
            tip="Move backward by 1 frame.",
            icon=os.path.sep.join([self._path, "icons", "backward.png"]),
            enabled=True,
            checkable=False,
            fun=self._backward_pressed,
        )
        self.backward_button.setAutoRepeat(True)
        commands_bar.addWidget(self.backward_button)

        # add the play/pause function
        self.play_button = self._command_button(
            tip="Play/Pause.",
            icon=os.path.sep.join([self._path, "icons", "play.png"]),
            enabled=True,
            checkable=False,
            fun=self._play_pressed,
        )
        commands_bar.addWidget(self.play_button)

        # add the move forward function
        self.forward_button = self._command_button(
            tip="Move forward by 1 frame.",
            icon=os.path.sep.join([self._path, "icons", "forward.png"]),
            enabled=True,
            checkable=False,
            fun=self._forward_pressed,
        )
        self.forward_button.setAutoRepeat(True)
        commands_bar.addWidget(self.forward_button)

        # speed controller
        self.speed_box = qtw.QSpinBox()
        self.speed_box.setFont(qtg.QFont("Arial", self._font_size))
        self.speed_box.setFixedHeight(self._button_size)
        self.speed_box.setFixedWidth(self._button_size * 2)
        self.speed_box.setToolTip("Speed control.")
        self.speed_box.setMinimum(1)
        self.speed_box.setMaximum(500)
        self.speed_box.setValue(100)
        self.speed_box.setSuffix("%")
        self.speed_box.setStyleSheet("border: none;")
        commands_bar.addWidget(self.speed_box)

        # add the loop function
        self.repeat_button = self._command_button(
            tip="Loop the frames.",
            icon=os.path.sep.join([self._path, "icons", "repeat.png"]),
            enabled=True,
            checkable=True,
            fun=self._loop_pressed,
        )
        commands_bar.addWidget(self.repeat_button)

        # add another separator
        commands_bar.addSeparator()

        # add the time label
        self.time_label = qtw.QLabel("00:00.000")
        self.time_label.setFont(qtg.QFont("Arial", self._font_size))
        self.time_label.setFixedHeight(self._button_size)
        self.time_label.setFixedWidth(self._button_size * 3)
        self.time_label.setAlignment(qtc.Qt.AlignCenter)
        commands_bar.addWidget(self.time_label)

        # add the time slider
        self.progress_slider = qtw.QSlider(qtc.Qt.Horizontal)
        self.progress_slider.setValue(0)
        self.progress_slider.setMinimum(0)
        self.progress_slider.setMaximum(len(self._times) - 1)
        self.progress_slider.setTickInterval(1)
        self.progress_slider.valueChanged.connect(self._slider_moved)
        self.progress_slider.setFixedHeight(self._button_size)
        commands_bar.addWidget(self.progress_slider)

        # set the timer for the player
        self._play_timer = qtc.QTimer()
        self._play_timer.timeout.connect(self._player)

        # add another separator
        commands_bar.addSeparator()

        # ground options
        self.ground_options = OptionGroup(
            label="Ground",
            min_value=0.1,
            max_value=2,
            step_value=0.1,
            font_size=self._font_size - 2,
            object_size=20,
            default_value=self._default_ground["linewidth"],
            default_color=self._default_ground["color"],
            default_zorder=self._default_ground["zorder"],
        )
        self.ground_options.valueChanged.connect(self._adjust_ground)
        self.ground_options.colorChanged.connect(self._adjust_ground)
        self.ground_options.zorderChanged.connect(self._adjust_ground)
        self.ground_options.setEnabled(self.ground_button.isEnabled())

        # marker options
        self.marker_options = OptionGroup(
            label="Markers",
            min_value=0.1,
            max_value=5,
            step_value=0.1,
            font_size=self._font_size - 2,
            object_size=20,
            default_value=self._default_marker["markersize"],
            default_color=self._default_marker["color"],
            default_zorder=self._default_marker["zorder"],
        )
        self.marker_options.valueChanged.connect(self._adjust_markers)
        self.marker_options.colorChanged.connect(self._adjust_markers)
        self.marker_options.zorderChanged.connect(self._adjust_markers)
        self.marker_options.setEnabled(self.marker_button.isEnabled())

        # force options
        self.force_options = OptionGroup(
            label="Forces",
            min_value=0.1,
            max_value=5,
            step_value=0.1,
            font_size=self._font_size - 2,
            object_size=20,
            default_value=self._default_force["linewidth"],
            default_color=self._default_force["color"],
            default_zorder=self._default_force["zorder"],
        )
        self.force_options.valueChanged.connect(self._adjust_forces)
        self.force_options.colorChanged.connect(self._adjust_forces)
        self.force_options.zorderChanged.connect(self._adjust_forces)
        self.force_options.setEnabled(self.force_button.isEnabled())

        # link options
        self.link_options = OptionGroup(
            label="Links",
            min_value=0.1,
            max_value=5,
            step_value=0.1,
            font_size=self._font_size - 2,
            object_size=20,
            default_value=self._default_link["linewidth"],
            default_color=self._default_link["color"],
            default_zorder=self._default_link["zorder"],
        )
        self.link_options.valueChanged.connect(self._adjust_links)
        self.link_options.colorChanged.connect(self._adjust_links)
        self.link_options.zorderChanged.connect(self._adjust_links)
        self.link_options.setEnabled(self.link_button.isEnabled())

        # text options
        self.text_options = OptionGroup(
            label="Labels",
            min_value=0.1,
            max_value=10,
            step_value=0.1,
            font_size=self._font_size - 2,
            object_size=20,
            default_value=self._default_text["size"],
            default_color=self._default_text["color"],
            default_zorder=self._default_text["zorder"],
        )
        self.text_options.valueChanged.connect(self._adjust_labels)
        self.text_options.colorChanged.connect(self._adjust_labels)
        self.text_options.zorderChanged.connect(self._adjust_labels)
        self.text_options.setEnabled(self.text_button.isEnabled())

        # reference options
        self.reference_options = OptionGroup(
            label="Reference frame",
            min_value=0.1,
            max_value=10,
            step_value=0.1,
            font_size=self._font_size - 2,
            object_size=20,
            default_value=self._default_ref["linewidth"],
            default_color=self._default_ref["color"],
            default_zorder=self._default_ref["zorder"],
        )
        self.reference_options.valueChanged.connect(self._adjust_references)
        self.reference_options.colorChanged.connect(self._adjust_references)
        self.reference_options.zorderChanged.connect(self._adjust_references)
        self.reference_options.setEnabled(self.reference_button.isEnabled())

        # emg signal options
        self.emg_signal_options = OptionGroup(
            label="EMG Signals",
            min_value=0.1,
            max_value=10,
            step_value=0.1,
            font_size=self._font_size - 2,
            object_size=20,
            default_value=self._default_emg_signal["linewidth"],
            default_color=self._default_emg_signal["color"],
            default_zorder=self._default_emg_signal["zorder"],
        )
        self.emg_signal_options.valueChanged.connect(self._adjust_emg_signals)
        self.emg_signal_options.colorChanged.connect(self._adjust_emg_signals)
        self.emg_signal_options.zorderChanged.connect(self._adjust_emg_signals)
        self.emg_signal_options.setEnabled(model.has_EmgSensor())

        # emg bar options
        self.emg_bar_options = OptionGroup(
            label="EMG Bars",
            min_value=0.1,
            max_value=10,
            step_value=0.1,
            font_size=self._font_size - 2,
            object_size=20,
            default_value=self._default_emg_bar["linewidth"],
            default_color=self._default_emg_bar["color"],
            default_zorder=self._default_emg_bar["zorder"],
        )
        self.emg_bar_options.valueChanged.connect(self._adjust_emg_bars)
        self.emg_bar_options.colorChanged.connect(self._adjust_emg_bars)
        self.emg_bar_options.zorderChanged.connect(self._adjust_emg_bars)
        self.emg_bar_options.setEnabled(model.has_EmgSensor())

        # options pane
        options_layout = qtw.QGridLayout()
        options_layout.setSpacing(10)
        options_layout.addWidget(self.marker_options, 1, 1)
        options_layout.addWidget(self.force_options, 2, 1)
        options_layout.addWidget(self.link_options, 3, 1)
        options_layout.addWidget(self.reference_options, 4, 1)
        options_layout.addWidget(self.text_options, 1, 2)
        options_layout.addWidget(self.emg_signal_options, 2, 2)
        options_layout.addWidget(self.emg_bar_options, 3, 2)
        options_layout.addWidget(self.ground_options, 4, 2)
        self.options_pane = qtw.QDialog(parent=self.option_button)
        self.options_pane.setWindowFlags(qtc.Qt.FramelessWindowHint)
        self.options_pane.setWindowModality(qtc.Qt.NonModal)
        self.options_pane.setLayout(options_layout)
        options_icon = os.path.sep.join([self._path, "icons", "options.png"])
        self.setWindowIcon(self._makeIcon(options_icon))

        # set the option pane button
        self.option_button = self._command_button(
            tip="Options.",
            icon=options_icon,
            enabled=True,
            checkable=True,
            fun=self._options_pressed,
        )
        self.option_button.setChecked(False)
        commands_bar.addWidget(self.option_button)

        # widget layout
        layout = qtw.QVBoxLayout()
        layout.addWidget(splitter)
        layout.addWidget(commands_bar)
        self.setLayout(layout)
        self.installEventFilter(self)
        self.setWindowTitle("Model3DWidget")
        icon_path = os.path.sep.join([self._path, "icons", "main.png"])
        self.setWindowIcon(self._makeIcon(icon_path))

        # set the starting view
        self._home_pressed()
        self._ground_button_pressed()
        self._update_figure()
        self.options_pane.show()
        self.options_pane.hide()
        self._adjust_emg_bars()
        self._adjust_emg_signals()
        self._adjust_forces()
        self._adjust_labels()
        self._adjust_markers()
        self._adjust_references()
        self._adjust_links()

    def _plotArrow(self, df, ax, **options):
        """
        plot a matplotlib arrow.

        Parameters
        ----------
        df: DataFrame
            a line containing all data.

        ax: matplotlib axis
            the axis on which the arrow has to be plotted.

        options: any
            additional rendering options passed to each element.

        Returns
        -------
        out: dict
            a dict containing all the axes.
        """
        p0x, p0y, p0z = df.values.flatten()[:3]
        p1x, p1y, p1z = df.values.flatten()[3:6]
        a0x, a0y, a0z = df.values.flatten()[6:9]
        a1x, a1y, a1z = df.values.flatten()[9:12]
        return {
            "Line": ax.plot(
                np.array([p0x, p1x]),
                np.array([p0y, p1y]),
                np.array([p0z, p1z]),
                animated=True,
                **options,
            )[0],
            "Arrow1": ax.plot(
                np.array([a0x, p1x]),
                np.array([a0y, p1y]),
                np.array([a0z, p1z]),
                animated=True,
                **options,
            )[0],
            "Arrow2": ax.plot(
                np.array([a1x, p1x]),
                np.array([a1y, p1y]),
                np.array([a1z, p1z]),
                animated=True,
                **options,
            )[0],
        }

    def _makeArrow(self, p0, p1, vert_axis):
        """
        private method used to obtain the data required to draw an arrow.

        Parameters
        ----------
        p0: DataFrame
            the coordinates of the origin.

        p1: DataFrame
            the coordinates of the end.

        vert_axis: int
            the number of the column in p0 and p1 corresponding to the
            vertical axis

        Returns
        -------
        arrow: DataFrame
            a dataframe containing the coordinates required to render
            the arrow.
        """
        xs = p0.index
        d0 = p0.values
        d1 = p1.values
        m = np.array([i == vert_axis for i in range(d0.shape[1])]).astype(int)
        m = m.flatten()
        off = np.sqrt(np.nansum((d1 - d0) ** 2, axis=1))
        off /= np.cos(np.deg2rad(self._arrow_angle))
        off = m * np.ones(p0.shape) * np.atleast_2d(off).T
        a0 = (d0 + off - d1) * self._arrow_length + d1
        a1 = (d0 - off - d1) * self._arrow_length + d1
        outs = []
        cols = ["X", "Y", "Z"]
        for v, l in zip([d0, d1, a0, a1], ["P0", "P1", "A0", "A1"]):
            ys = pd.MultiIndex.from_tuples([(l, c) for c in cols])
            outs += [pd.DataFrame(v, index=xs, columns=ys)]
        return pd.concat(outs, axis=1)

    def _makeIcon(self, file):
        """
        internal function used to build icons from png file.
        """
        pix = qtg.QPixmap(file).scaled(self._button_size, self._button_size)
        return qtg.QIcon(pix)

    def eventFilter(self, source, event):
        """
        handle events
        """
        if self.option_button.isChecked():
            self._adjust_options_pane_position()
            self.options_pane.setFocus()
            self.options_pane.activateWindow()
        return super().eventFilter(source, event)

    def is_running(self):
        """
        check if the player is running.
        """
        return self._is_running

    def _command_button(self, tip, icon, enabled, checkable, fun):
        """
        private method used to generate valid buttons for the command bar.

        Parameters
        ----------
        icon: str
            the path to the image used for the button

        tip: str
            a tip appearing pointing over the action

        enabled: bool
            should the button be enabled?

        checkable: bool
            should the button be checkable

        fun: function
            the button press handler.

        Returns
        -------
        obj: qtw.QPushButton
            a novel PushButton object.
        """
        button = qtw.QPushButton()
        button.setFlat(True)
        button.setToolTip(tip)
        button.setEnabled(enabled)
        button.setCheckable(checkable)
        button.setFixedHeight(self._button_size)
        button.setFixedWidth(self._button_size)
        if checkable and enabled:
            button.setChecked(True)
        if fun is not None:
            button.clicked.connect(fun)
        if icon is not None:
            button.setIcon(self._makeIcon(icon))
        return button

    def _move_forward(self):
        """
        function handling the press of the play button.
        """
        frame = self.progress_slider.value()
        next_frame = frame + 1
        if next_frame > self.progress_slider.maximum():
            if self.repeat_button.isChecked():
                next_frame = 0
            else:
                next_frame = self.progress_slider.maximum()
        self.progress_slider.setValue(next_frame)

    def _move_backward(self):
        """
        function handling the press of the play button.
        """
        frame = self.progress_slider.value()
        next_frame = frame - 1
        if next_frame < 0:
            if self.repeat_button.isChecked():
                next_frame = self.progress_slider.maximum()
            else:
                next_frame = 0
        self.progress_slider.setValue(next_frame)

    def _start_player(self):
        """
        stop the player
        """
        self._play_start_time = get_time()
        self._play_timer.start(self._update_rate)
        self._is_running = True
        icon_path = os.path.sep.join([self._path, "icons", "pause.png"])
        pxmap = qtg.QPixmap(icon_path)
        pxmap = pxmap.scaled(self._button_size, self._button_size)
        icon = qtg.QIcon(pxmap)
        self.play_button.setIcon(icon)
        self.play_button.setStatusTip("Pause.")

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
        self.play_button.setStatusTip("Play.")

    def _player(self):
        """
        player event handler
        """
        lapsed = (get_time() - self._play_start_time) * 1000 + self._times[0]
        speed = float(self.speed_box.text()[:-1]) / 100
        lapsed = lapsed * speed
        if lapsed > self._times[-1] - self._times[0]:
            if self.repeat_button.isChecked():
                self._play_start_time = get_time()
                self.progress_slider.setValue(0)
            else:
                self._stop_player()
                self.progress_slider.setValue(self.progress_slider.maximum())
        else:
            self.progress_slider.setValue(np.argmin(abs(self._times - lapsed)))

    def _slider_moved(self):
        """
        event handler for the slider value update.
        """
        # handle the options
        self.option_button.setChecked(False)
        self.options_pane.hide()

        # handle the slider
        self._actual_frame = self.progress_slider.value()
        self._update_figure()

    def _resize_event(self, event):
        """
        handler for a figure resize event.
        """
        if self._figure3D is not None:
            self._figure3D.tight_layout()
            self._figure3D.canvas.draw()

        if self._figureEMG is not None:
            self._figureEMG.tight_layout()
            self._figureEMG.canvas.draw()

    def _update_force_alpha(self, label, alpha):
        """
        update the force alpha values.

        Parameters
        ----------
        label: str
            the name of the object to be updated.

        alpha: float
            the alpha value to be updated.

        has_text: bool
            should also the text value be updated?
        """
        if self.force_button.isChecked():
            f_alpha = self.force_options.color()[-1]
            if self.text_button.isChecked():
                t_alpha = self.text_options.color()[-1]
            else:
                t_alpha = 0
        else:
            f_alpha = 0
            t_alpha = 0
        self._ForcePlatform3D[label]["Line"]._alpha = alpha * f_alpha
        self._ForcePlatform3D[label]["Arrow1"]._alpha = alpha * f_alpha
        self._ForcePlatform3D[label]["Arrow2"]._alpha = alpha * f_alpha
        self._ForcePlatform3D[label]["Point"]._alpha = alpha * f_alpha
        self._ForcePlatform3D[label]["Label"]._alpha = alpha * t_alpha

    def _update_link_alpha(self, label, alpha):
        """
        update the link alpha values.

        Parameters
        ----------
        label: str
            the name of the object to be updated.

        alpha: float
            the alpha value to be updated.

        has_text: bool
            should also the text value be updated?
        """
        if self.link_button.isChecked():
            link_alpha = self.link_options.color()[-1]
            self._Link3D[label]["Line"]._alpha = alpha * link_alpha
        else:
            self._Link3D[label]["Line"]._alpha = 0

    def _update_marker_alpha(self, label, alpha):
        """
        update the marker alpha values.

        Parameters
        ----------
        label: str
            the name of the object to be updated.

        alpha: float
            the alpha value to be updated.

        has_text: bool
            should also the text value be updated?
        """
        if self.marker_button.isChecked():
            marker_alpha = self.marker_options.color()[-1]
            self._Marker3D[label]["Point"]._alpha = alpha * marker_alpha
            if self.text_button.isChecked():
                text_alpha = self.text_options.color()[-1]
                self._Marker3D[label]["Label"]._alpha = alpha * text_alpha
            else:
                self._Marker3D[label]["Label"]._alpha = 0
        else:
            self._Marker3D[label]["Label"]._alpha = 0
            self._Marker3D[label]["Point"]._alpha = 0

    def _update_figure(self):
        """
        update the actual rendered figure.
        """

        # update the timer
        time = self._times[self._actual_frame]
        minutes = time // 60000
        seconds = (time - minutes * 60000) // 1000
        msec = time - minutes * 60000 - seconds * 1000
        lbl = "{:02d}:{:02d}.{:03d}".format(minutes, seconds, msec)
        self.time_label.setText(lbl)

        # update the data
        for sns, dcts in self.data.items():
            for t, df in dcts.items():

                if sns == "ForcePlatform3D":
                    if time in df.index.to_numpy():
                        v = df.loc[time].values
                        x0, y0, z0, x1, y1, z1 = v[:6]
                        ax0, ay0, az0, ax1, ay1, az1, s = v[6:]
                        self._ForcePlatform3D[t]["Line"].set_data_3d(
                            np.array([x0, x1]),
                            np.array([y0, y1]),
                            np.array([z0, z1]),
                        )
                        self._ForcePlatform3D[t]["Arrow1"].set_data_3d(
                            np.array([ax0, x1]),
                            np.array([ay0, y1]),
                            np.array([az0, z1]),
                        )
                        self._ForcePlatform3D[t]["Arrow2"].set_data_3d(
                            np.array([ax1, x1]),
                            np.array([ay1, y1]),
                            np.array([az1, z1]),
                        )
                        self._ForcePlatform3D[t]["Point"].set_data_3d(
                            np.array([x0]),
                            np.array([y0]),
                            np.array([z0]),
                        )
                        self._ForcePlatform3D[t]["Label"]._x = x0
                        self._ForcePlatform3D[t]["Label"]._y = y0
                        self._ForcePlatform3D[t]["Label"]._z = z0
                        self._update_force_alpha(t, s)
                    else:
                        self._update_force_alpha(t, 0)

                elif sns == "Link3D":
                    if time in df.index.to_numpy():
                        x, y, z, u, v, w, s = df.loc[time].values
                        self._Link3D[t]["Line"].set_data_3d(
                            np.array([x, u]),
                            np.array([y, v]),
                            np.array([z, w]),
                        )
                        self._update_link_alpha(t, s)
                    else:
                        self._update_link_alpha(t, 0)

                elif sns == "Marker3D":
                    if time in df.index.to_numpy():
                        x, y, z, s = df.loc[time].values
                        self._Marker3D[t]["Point"].set_data_3d(x, y, z)
                        self._Marker3D[t]["Label"]._x = x
                        self._Marker3D[t]["Label"]._y = y
                        self._Marker3D[t]["Label"]._z = z
                        self._update_marker_alpha(t, s)
                    else:
                        self._update_marker_alpha(t, 0)

                elif sns == "EmgSensor":
                    self._EmgSensor[t]["Bar"].set_xdata([time, time])

        # update the figures
        if self._figure3D is not None:
            self._FigureAnimator3D.update()

        if self._figureEMG is not None:
            self._FigureAnimatorEMG.update()

    def _reference_pressed(self):
        """
        handler for the reference button.
        """
        # handle the options
        self.option_button.setChecked(False)
        self.options_pane.hide()

        # update the reference frame
        alpha = self.reference_options.color()[-1]
        a = alpha if self.reference_button.isChecked() else 0
        for n in self._ReferenceFrame3D:
            for l in ["Line", "Arrow1", "Arrow2", "Label", "Point"]:
                self._ReferenceFrame3D[n][l].set_alpha(a)
        self._update_figure()

    def _play_pressed(self):
        """
        method handling the play button press events.
        """
        # handle the options
        self.option_button.setChecked(False)
        self.options_pane.hide()

        # handle the player
        if self.is_running():
            self._stop_player()
        else:
            self._start_player()

    def _forward_pressed(self):
        """
        method handling the forward button press events.
        """
        # handle the options
        self.option_button.setChecked(False)
        self.options_pane.hide()

        # handle the button
        self._stop_player()
        self._move_forward()

    def _backward_pressed(self):
        """
        method handling the forward button press events.
        """
        # handle the options
        self.option_button.setChecked(False)
        self.options_pane.hide()

        # handle the button
        self._stop_player()
        self._move_backward()

    def _home_pressed(self):
        """
        method handling the home button press events.
        """
        # handle the options
        self.option_button.setChecked(False)
        self.options_pane.hide()

        # handle the button
        self._stop_player()
        self._axis3D.elev = self._default_view["elev"]
        self._axis3D.azim = self._default_view["azim"]
        self._axis3D.set_xlim(*self._limits)
        self._axis3D.set_ylim(*self._limits)
        self._axis3D.set_zlim(*self._limits)
        self.progress_slider.setValue(0)
        self._figure3D.canvas.draw()
        self._figureEMG.canvas.draw()

    def _media_action_pressed(self):
        """
        handle the pressure of the media action button.
        """
        self.option_button.setChecked(False)
        self.options_pane.hide()

    def _options_pressed(self):
        """
        method handling the options button press events.
        """
        self._stop_player()
        self.options_pane.setVisible(self.option_button.isChecked())

    def _loop_pressed(self):
        """
        method handling the loop button press events.
        """
        self.option_button.setChecked(False)
        self.options_pane.hide()

    def _marker_button_pressed(self):
        """
        handle the marker button interaction.
        """
        # handle the options
        self.option_button.setChecked(False)
        self.options_pane.hide()

        # update the figure
        self._update_figure()

    def _force_button_pressed(self):
        """
        handle the force button interaction.
        """
        # handle the options
        self.option_button.setChecked(False)
        self.options_pane.hide()

        # update the figure
        self._update_figure()

    def _link_button_pressed(self):
        """
        handle the link button interaction.
        """
        # handle the options
        self.option_button.setChecked(False)
        self.options_pane.hide()

        # update the figure
        self._update_figure()

    def _text_button_pressed(self):
        """
        handle the text button interaction.
        """
        # handle the options
        self.option_button.setChecked(False)
        self.options_pane.hide()

        # update the figure
        self._update_figure()

    def _ground_button_pressed(self):
        """
        handle the ground button interaction.
        """

        # update the alpha
        if self.ground_button.isChecked():
            alpha = self.ground_options.color()[-1]
        else:
            alpha = 0
        for i in self._Ground3D:
            self._Ground3D[i]["Grid"].set_alpha(alpha)

        # update the figure
        self._update_figure()

    def _adjust_options_pane_position(self):
        """
        method handling the location of the options pane
        """
        butRect = self.option_button.rect()
        cntRect = self.options_pane.rect()
        loc = butRect.topRight()
        loc -= cntRect.bottomRight()
        loc = self.option_button.mapToGlobal(loc)
        self.options_pane.move(loc)

    def _adjust_speed(self):
        """
        adjust the player speed.
        """
        # handle the options
        self.option_button.setChecked(False)
        self.options_pane.hide()

        # adjust the speed
        self.speed_box.setValue(self.speed_slider.value())

    def _adjust_ground(self):
        """
        adjust the ground appearance.
        """
        c = self.ground_options.color()
        s = self.ground_options.size()
        z = self.ground_options.zorder()
        self._Ground3D["Ground"]["Grid"].zorder = z
        self._Ground3D["Ground"]["Grid"].set_color(c)
        self._Ground3D["Ground"]["Grid"].linewidth = s / 10
        self._update_figure()

    def _adjust_markers(self):
        """
        adjust the markers appearance.
        """
        c = self.marker_options.color()
        s = self.marker_options.size()
        z = self.marker_options.zorder()
        for n in self._Marker3D:
            self._Marker3D[n]["Point"].set_color(c)
            self._Marker3D[n]["Point"].set_ms(s)
            self._Marker3D[n]["Point"].zorder = z
        for n in self._ForcePlatform3D:
            self._ForcePlatform3D[n]["Point"].set_ms(s)
        for n in self._ReferenceFrame3D:
            self._ReferenceFrame3D[n]["Point"].set_ms(s)
        self._update_figure()

    def _adjust_forces(self):
        """
        adjust the forces appearance.
        """
        c = self.force_options.color()
        s = self.force_options.size()
        z = self.force_options.zorder()
        for n in self._ForcePlatform3D:
            for k in ["Line", "Arrow1", "Arrow2", "Point"]:
                self._ForcePlatform3D[n][k].set_color(c)
                self._ForcePlatform3D[n][k].zorder = z
                if k != "Point":
                    self._ForcePlatform3D[n][k].set_linewidth(s)
        self._update_figure()

    def _adjust_links(self):
        """
        adjust the links appearance.
        """
        for n in self._Link3D:
            self._Link3D[n]["Line"].set_color(self.link_options.color())
            self._Link3D[n]["Line"].set_linewidth(self.link_options.size())
            self._Link3D[n]["Line"].zorder = self.link_options.zorder()
        self._update_figure()

    def _adjust_references(self):
        """
        adjust the reference frame appearance.
        """
        c = self.reference_options.color()
        s = self.reference_options.size()
        z = self.reference_options.zorder()
        for n in self._ReferenceFrame3D:
            for a in ["Line", "Arrow1", "Arrow2", "Label", "Point"]:
                self._ReferenceFrame3D[n][a].set_color(c)
                self._ReferenceFrame3D[n][a].zorder = z
                if a not in ["Label", "Point"]:
                    self._ReferenceFrame3D[n][a].set_linewidth(s)
                if not self.reference_button.isChecked():
                    self._ReferenceFrame3D[n][a].set_alpha(0)
        self._update_figure()

    def _adjust_labels(self):
        """
        adjust the text appearance.
        """
        s = self.text_options.size()
        c = self.text_options.color()
        z = self.text_options.zorder()
        for n in self._Marker3D:
            self._Marker3D[n]["Label"].set_size(s)
            self._Marker3D[n]["Label"].set_color(c)
            self._Marker3D[n]["Label"].zorder = z
        for n in self._ForcePlatform3D:
            self._ForcePlatform3D[n]["Label"].set_size(s)
            self._ForcePlatform3D[n]["Label"].set_color(c)
            self._ForcePlatform3D[n]["Label"].zorder = z
        for n in self._ReferenceFrame3D:
            self._ReferenceFrame3D[n]["Label"].set_size(s)
        self._update_figure()

    def _adjust_emg_signals(self):
        """
        adjust the emg signals appearance.
        """
        col = self.emg_signal_options.color()
        val = self.emg_signal_options.size()
        order = self.emg_signal_options.zorder()
        for n in self._EmgSensor:
            self._EmgSensor[n]["Signal"].set_color(col)
            self._EmgSensor[n]["Signal"].set_linewidth(val)
            self._EmgSensor[n]["Signal"].zorder = order
        self._update_figure()

    def _adjust_emg_bars(self):
        """
        adjust the emg bars appearance.
        """
        col = self.emg_bar_options.color()
        val = self.emg_bar_options.size()
        order = self.emg_bar_options.zorder()
        for n in self._EmgSensor:
            self._EmgSensor[n]["Bar"].set_color(col)
            self._EmgSensor[n]["Bar"].set_linewidth(val)
            self._EmgSensor[n]["Bar"].zorder = order
        self._update_figure()

    def closeEvent(self, event):
        self.close()
