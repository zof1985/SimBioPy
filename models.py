# MODELS MODULE


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
from .utils import *
from .geometry import *
from .sensors import *


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


class OptionPane(qtw.QWidget):
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
        zorder=0,
        min_value=1,
        max_value=100,
        step_value=1,
        default_value=1,
        default_color=(255, 0, 0, 255),
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

        # label
        self.label = qtw.QLabel(label)
        label_font = qtg.QFont("Arial", self.font_size)
        label_font.setBold(True)
        self.label.setFont(label_font)
        self.label.setFixedHeight(self.object_size)
        self.label.setAlignment(qtc.Qt.AlignLeft | qtc.Qt.AlignBottom)

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
        self.zorderBox.setValue(zorder)
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
        data_layout = qtw.QHBoxLayout()
        data_layout.addWidget(color_widget)
        data_layout.addWidget(size_widget2)
        data_layout.addWidget(zorder_widget)
        data_layout.setSpacing(10)
        data_layout.setContentsMargins(0, 0, 0, 0)
        data_widget = qtw.QWidget()
        data_widget.setLayout(data_layout)
        layout = qtw.QVBoxLayout()
        layout.setSpacing(0)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.label)
        layout.addWidget(data_widget)
        self.setLayout(layout)

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
    _Text3D = {}
    _ReferenceFrame3D = {}
    _EmgSensor = {}
    _force3D_sclr = 1
    _actual_frame = None
    _FigureAnimator3D = None
    _FigureAnimatorEMG = None
    _play_start_time = None
    _path = None
    _init_view = {"elev": 10, "azim": 45, "vertical_axis": 2}
    _limits = None
    _marker_color = (1, 0, 0, 1)
    _force_color = (0, 1, 0, 0.7)
    _link_color = (0, 0, 1, 0.7)
    _text_color = (0, 0, 0, 0.5)
    _reference_color = (0, 0.5, 0.5, 0.5)
    _emg_signal_color = (0.5, 0.5, 0, 1)
    _emg_bar_color = (0.5, 0, 0.5, 1)

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

        # get the time indices
        df0 = model.stack()
        df3d = {"Sensor": ["Marker3D", "ForcePlatform3D", "Link3D"]}
        df3d = df0.loc[df0.isin(df3d).any(1)]
        self._times = np.unique(df3d["Time"].values.flatten())

        # get the data
        dfs = {}
        for sns in np.unique(df0["Sensor"].values.flatten()):
            dd = df0.loc[df0.isin([sns]).any(1)]

            if sns == "ForcePlatform3D":
                ori = dd.loc[dd.isin(["origin"]).any(1)]
                ori = ori.pivot(["Time", "Label"], "Dimension", "Amplitude")
                amp = dd.loc[dd.isin(["amplitude"]).any(1)]
                amp = amp.pivot(["Time", "Label"], "Dimension", "Amplitude")
                amp.columns = pd.Index(["U", "V", "W"])
                out = pd.concat([ori, amp], axis=1)

            elif sns == "Link3D":
                out = []
                for l in ["p0", "p1"]:
                    pp = dd.loc[dd.isin([l]).any(1)]
                    pp = pp.pivot(["Time", "Label"], "Dimension", "Amplitude")
                    cols = pp.columns.to_numpy()
                    idx = ["{}{}".format(i, l[1]) for i in cols]
                    pp.columns = pd.Index(idx)
                    out += [pp]
                out = pd.concat(out, axis=1)

            elif sns == "Marker3D":
                out = dd.pivot(["Time", "Label"], "Dimension", "Amplitude")

            elif sns == "EmgSensor":
                out = dd.pivot(["Time", "Label"], "Dimension", "Amplitude")
                time = np.atleast_2d(out.index.to_frame()["Time"].values).T
                out = pd.DataFrame(
                    data=np.hstack([time, time]),
                    columns=["X0", "X1"],
                    index=out.index,
                )

            # adjust the nans
            nans = np.any(np.isnan(out.values), 1)
            out.loc[nans] = 0
            alphas = np.ones((out.shape[0], 1))
            alphas[nans] = 0
            out.insert(out.shape[1], "Alpha", alphas)
            ix = out.index.to_frame()
            out.insert(0, "Label", ix["Label"].values.flatten())
            out.index = pd.Index(ix["Time"].values.flatten())

            # separate by label
            dd = {}
            for l in np.unique(out["Label"].values.flatten()):
                dd[l] = out.loc[out.isin([l]).any(1)]
                dd[l] = dd[l].drop("Label", axis=1)
            dfs[sns] = dd

        # check input data
        if len(dfs) == 0:
            raise ValueError("No valid data have been found in 'model'.")
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

        # check the vertical axis input
        if any([i == "Marker3D" for i in list(dfs.keys())]):
            dim = list(dfs["Marker3D"].values())[0].columns.to_numpy()
        else:
            dim = list(dfs["ForcePlatform3D"].values())[0].columns.to_numpy()
        txt = "vertical_axis not found in model."
        assert vertical_axis.upper() in dim, txt
        self._init_view["vertical_axis"] = vertical_axis.lower()

        # create the 3D model pane
        if model.has_ForcePlatform3D() or model.has_Marker3D():

            # generate the axis
            self._figure3D = pl.figure(dpi=self._dpi)
            self.canvas3D = FigureCanvasQTAgg(self._figure3D)
            self._axis3D = self._figure3D.add_subplot(
                projection="3d",
                proj_type="ortho",  # 'persp'
                adjustable="box",  # 'datalim'
                facecolor=(1, 1, 1, 0),
                frame_on=False,
            )

            # set the view
            self._axis3D.view_init(**self._init_view)

            # resizing event handler
            self._figure3D.canvas.mpl_connect(
                "resize_event",
                self._resize_event,
            )

            # make the panes transparent
            self._axis3D.xaxis.set_pane_color((1, 1, 1, 0))
            self._axis3D.yaxis.set_pane_color((1, 1, 1, 0))
            self._axis3D.zaxis.set_pane_color((1, 1, 1, 0))

            # make the axis lines transparent
            self._axis3D.xaxis.line.set_color((1, 1, 1, 0))
            self._axis3D.yaxis.line.set_color((1, 1, 1, 0))
            self._axis3D.zaxis.line.set_color((1, 1, 1, 0))

            # remove the ticks
            self._axis3D.xaxis.set_ticks([])
            self._axis3D.yaxis.set_ticks([])
            self._axis3D.zaxis.set_ticks([])

            # make the grid lines transparent
            self._axis3D.xaxis._axinfo["grid"]["color"] = (1, 1, 1, 0)
            self._axis3D.yaxis._axinfo["grid"]["color"] = (1, 1, 1, 0)
            self._axis3D.zaxis._axinfo["grid"]["color"] = (1, 1, 1, 0)

            # set the axis limits
            if model.has_Marker3D():
                edges = [v.values for v in self.data["Marker3D"].values()]
            else:
                edges = [v for v in self.data["ForcePlatform3D"].values()]
                edges = [v.values[:, :3] for v in edges]
            edges = np.concatenate(edges, axis=0)
            maxc = max(0, np.nanmax(edges) * 1.5)
            minc = min(0, np.nanmin(edges) * 1.5)
            self._axis3D.set_xlim(minc, maxc)
            self._axis3D.set_ylim(minc, maxc)
            self._axis3D.set_zlim(minc, maxc)

            # set the initial limits
            self._limits = (minc, maxc)

            # plot the reference frame
            self._ReferenceFrame3D = {
                "X": {
                    "Versor": self._axis3D.quiver(
                        0,
                        0,
                        0,
                        0.15 * (maxc - minc),
                        0,
                        0,
                        color="gold",
                        animated=True,
                        length=1,
                        linewidth=0.5,
                        zorder=3,
                    ),
                    "Text": self._axis3D.text(
                        0.2 * (maxc - minc),
                        0,
                        0,
                        "X",
                        size=3,
                        color="red",
                        animated=True,
                        zorder=2,
                        ha="center",
                        va="center",
                    ),
                },
                "Y": {
                    "Versor": self._axis3D.quiver(
                        0,
                        0,
                        0,
                        0,
                        0.15 * (maxc - minc),
                        0,
                        color="gold",
                        animated=True,
                        length=1,
                        linewidth=0.5,
                        zorder=3,
                    ),
                    "Text": self._axis3D.text(
                        0,
                        0.2 * (maxc - minc),
                        0,
                        "Y",
                        size=3,
                        color="red",
                        animated=True,
                        zorder=2,
                        ha="center",
                        va="center",
                    ),
                },
                "Z": {
                    "Versor": self._axis3D.quiver(
                        0,
                        0,
                        0,
                        0,
                        0,
                        0.15 * (maxc - minc),
                        color="gold",
                        animated=True,
                        length=1,
                        linewidth=0.5,
                        zorder=3,
                    ),
                    "Text": self._axis3D.text(
                        0,
                        0,
                        0.2 * (maxc - minc),
                        "Z",
                        size=3,
                        color="red",
                        animated=True,
                        zorder=2,
                        ha="center",
                        va="center",
                    ),
                },
            }

            # force amplitude scaler
            if model.has_ForcePlatform3D():
                max_range = maxc - minc
                self._force3D_sclr = max_range / maxc

            # set the objects
            for sns in self.data:
                for l in self.data[sns]:

                    if sns == "Marker3D":
                        self._Marker3D[l] = self._axis3D.plot(
                            0,
                            0,
                            0,
                            marker="o",
                            alpha=0.5,
                            markersize=2,
                            color="navy",
                            animated=True,
                            zorder=2,
                        )[0]
                        self._Text3D[l] = self._axis3D.text(
                            0,
                            0,
                            0,
                            "{:25s}".format(l),
                            alpha=1.0,
                            size=3,
                            animated=True,
                            zorder=1,
                        )

                    elif sns == "ForcePlatform3D":
                        self._ForcePlatform3D[l] = self._axis3D.quiver(
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            alpha=0.5,
                            color="darkgreen",
                            animated=True,
                            zorder=3,
                        )[0]
                        self._Text3D[l] = self._axis3D.text(
                            0,
                            0,
                            0,
                            "{:25s}".format(l),
                            alpha=1.0,
                            size=3,
                            animated=True,
                            zorder=1,
                        )

                    elif sns == "Link3D":
                        self._Link3D[l] = self._axis3D.plot(
                            np.array([0, 0]),
                            np.array([0, 0]),
                            np.array([0, 0]),
                            alpha=0.3,
                            color="darkred",
                            linewidth=1,
                            animated=True,
                            zorder=3,
                        )[0]

            # setup the Figure Animator
            artists = [i for i in self._Marker3D.values()]
            artists += [i for i in self._ForcePlatform3D.values()]
            artists += [i for i in self._Text3D.values()]
            artists += [i for i in self._Link3D.values()]
            for ax in self._ReferenceFrame3D.values():
                artists += [val for val in ax.values()]
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

        # add a separator
        commands_bar.addSeparator()

        # function show/hide markers
        self.marker_button = self._command_button(
            tip="Show/Hide the Marker3D objects.",
            icon=os.path.sep.join([self._path, "icons", "markers.png"]),
            enabled=model.has_Marker3D(),
            checkable=True,
            fun=self._update_figure,
        )
        commands_bar.addWidget(self.marker_button)

        # function show/hide forces function
        self.force_button = self._command_button(
            tip="Show/Hide the ForcePlatform3D objects.",
            icon=os.path.sep.join([self._path, "icons", "forces.png"]),
            enabled=model.has_ForcePlatform3D(),
            checkable=True,
            fun=self._update_figure,
        )
        commands_bar.addWidget(self.force_button)

        # function show/hide links function
        self.link_button = self._command_button(
            tip="Show/Hide the Link3D objects.",
            icon=os.path.sep.join([self._path, "icons", "links.png"]),
            enabled=model.has_Link3D(),
            checkable=True,
            fun=self._update_figure,
        )
        commands_bar.addWidget(self.link_button)

        # function show/hide labels function
        self.text_button = self._command_button(
            tip="Show/Hide the labels.",
            icon=os.path.sep.join([self._path, "icons", "txt.png"]),
            enabled=model.has_ForcePlatform3D() | model.has_Marker3D(),
            checkable=True,
            fun=self._update_figure,
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

        # add another separator
        commands_bar.addSeparator()

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
            fun=None,
        )
        commands_bar.addWidget(self.repeat_button)

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

        # marker options
        self.marker_options = OptionPane(
            label="Markers",
            min_value=0.1,
            max_value=10,
            step_value=0.1,
            default_value=1,
            default_color=self._marker_color,
            font_size=self._font_size - 2,
            object_size=20,
        )
        self.marker_options.valueChanged.connect(self._adjust_markers)
        self.marker_options.colorChanged.connect(self._adjust_markers)
        self.marker_options.zorderChanged.connect(self._adjust_markers)
        self.marker_options.setEnabled(self.marker_button.isEnabled())

        # force options
        self.force_options = OptionPane(
            label="Forces",
            min_value=0.1,
            max_value=10,
            step_value=0.1,
            default_value=0.5,
            default_color=self._force_color,
            font_size=self._font_size - 2,
            object_size=20,
        )
        self.force_options.valueChanged.connect(self._adjust_forces)
        self.force_options.colorChanged.connect(self._adjust_forces)
        self.force_options.zorderChanged.connect(self._adjust_forces)
        self.force_options.setEnabled(self.force_button.isEnabled())

        # link options
        self.link_options = OptionPane(
            label="Links",
            min_value=0.1,
            max_value=10,
            step_value=0.1,
            default_value=0.5,
            default_color=self._link_color,
            font_size=self._font_size - 2,
            object_size=20,
        )
        self.link_options.valueChanged.connect(self._adjust_links)
        self.link_options.colorChanged.connect(self._adjust_links)
        self.link_options.zorderChanged.connect(self._adjust_links)
        self.link_options.setEnabled(self.link_button.isEnabled())

        # text options
        self.text_options = OptionPane(
            label="Labels",
            min_value=0.1,
            max_value=10,
            step_value=0.1,
            default_value=3,
            default_color=self._text_color,
            font_size=self._font_size - 2,
            object_size=20,
        )
        self.text_options.valueChanged.connect(self._adjust_labels)
        self.text_options.colorChanged.connect(self._adjust_labels)
        self.text_options.zorderChanged.connect(self._adjust_labels)
        self.text_options.setEnabled(self.text_button.isEnabled())

        # reference options
        self.reference_options = OptionPane(
            label="Reference frame",
            min_value=0.1,
            max_value=10,
            step_value=0.1,
            default_value=1,
            default_color=self._reference_color,
            font_size=self._font_size - 2,
            object_size=20,
        )
        self.reference_options.valueChanged.connect(self._adjust_references)
        self.reference_options.colorChanged.connect(self._adjust_references)
        self.reference_options.zorderChanged.connect(self._adjust_references)
        self.reference_options.setEnabled(self.reference_button.isEnabled())

        # emg signal options
        self.emg_signal_options = OptionPane(
            label="EMG Signals",
            min_value=0.1,
            max_value=10,
            step_value=0.1,
            default_value=0.5,
            default_color=self._emg_signal_color,
            font_size=self._font_size - 2,
            object_size=20,
        )
        self.emg_signal_options.valueChanged.connect(self._adjust_emg_signals)
        self.emg_signal_options.colorChanged.connect(self._adjust_emg_signals)
        self.emg_signal_options.zorderChanged.connect(self._adjust_emg_signals)
        self.emg_signal_options.setEnabled(model.has_EmgSensor())

        # emg bar options
        self.emg_bar_options = OptionPane(
            label="EMG Bars",
            min_value=0.1,
            max_value=10,
            step_value=0.1,
            default_value=0.3,
            default_color=self._emg_bar_color,
            font_size=self._font_size - 2,
            object_size=20,
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
        self.options_pane = qtw.QWidget(parent=self.option_button)
        self.options_pane.setWindowFlags(qtc.Qt.FramelessWindowHint)
        self.options_pane.setWindowModality(qtc.Qt.NonModal)
        self.options_pane.setLayout(options_layout)

        # set the option pane button
        self.option_button = self._command_button(
            tip="Options.",
            icon=os.path.sep.join([self._path, "icons", "options.png"]),
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

        # set the starting view
        self._home_pressed()
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
        if icon is not None:
            icon = qtg.QPixmap(icon)
            icon = icon.scaled(self._button_size, self._button_size)
            button.setIcon(icon)
        button.setToolTip(tip)
        button.setEnabled(enabled)
        button.setCheckable(checkable)
        if checkable and enabled:
            button.setChecked(True)
        if fun is not None:
            button.clicked.connect(fun)
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
                    x, y, z, u, v, w, s = df.loc[time].values
                    self._ForcePlatform3D[t].set_data_3d(x, y, z, u, v, w)
                    self._Text3D[t]._x = x
                    self._Text3D[t]._y = y
                    self._Text3D[t]._z = z
                    if self.force_button.isChecked():
                        force_alpha = self.force_options.color()[-1]
                        self._ForcePlatform3D[t]._alpha = s * force_alpha
                        if self.text_button.isChecked():
                            text_alpha = self.text_options.color()[-1]
                            self._Text3D[t]._alpha = s * text_alpha
                        else:
                            self._Text3D[t]._alpha = 0
                    else:
                        self._ForcePlatform3D[t]._alpha = 0
                        self._Text3D[t]._alpha = 0

                elif sns == "Link3D":
                    x, y, z, u, v, w, s = df.loc[time].values
                    self._Link3D[t].set_data_3d(
                        np.array([x, u]),
                        np.array([y, v]),
                        np.array([z, w]),
                    )
                    if self.link_button.isChecked():
                        link_alpha = self.link_options.color()[-1]
                        self._Link3D[t]._alpha = s * link_alpha
                    else:
                        self._Link3D[t]._alpha = 0

                elif sns == "Marker3D":
                    x, y, z, s = df.loc[time].values
                    self._Marker3D[t].set_data_3d(x, y, z)
                    self._Text3D[t]._x = x
                    self._Text3D[t]._y = y
                    self._Text3D[t]._z = z
                    if self.marker_button.isChecked():
                        marker_alpha = self.marker_options.color()[-1]
                        self._Marker3D[t]._alpha = s * marker_alpha
                        if self.text_button.isChecked():
                            text_alpha = self.text_options.color()[-1]
                            self._Text3D[t]._alpha = s * text_alpha
                        else:
                            self._Text3D[t]._alpha = 0
                    else:
                        self._Marker3D[t]._alpha = 0
                        self._Text3D[t]._alpha = 0

                elif sns == "EmgSensor":
                    x, y, s = df.loc[time].values
                    self._EmgSensor[t]["Bar"].set_xdata([x, y])

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

        # update the reference frame
        alpha = self.reference_options.color()[-1]
        for n in self._ReferenceFrame3D:
            if self.reference_button.isChecked():
                self._ReferenceFrame3D[n]["Text"].set_alpha(alpha)
                self._ReferenceFrame3D[n]["Versor"].set_alpha(alpha)
            else:
                self._ReferenceFrame3D[n]["Text"].set_alpha(0)
                self._ReferenceFrame3D[n]["Versor"].set_alpha(0)
        self._update_figure()

    def _play_pressed(self):
        """
        method handling the play button press events.
        """
        # handle the options
        self.option_button.setChecked(False)

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

        # handle the button
        self._stop_player()
        self._move_forward()

    def _backward_pressed(self):
        """
        method handling the forward button press events.
        """
        # handle the options
        self.option_button.setChecked(False)

        # handle the button
        self._stop_player()
        self._move_backward()

    def _home_pressed(self):
        """
        method handling the home button press events.
        """
        # handle the options
        self.option_button.setChecked(False)

        # handle the button
        self._stop_player()
        self._axis3D.elev = self._init_view["elev"]
        self._axis3D.azim = self._init_view["azim"]
        self._axis3D.set_xlim(*self._limits)
        self._axis3D.set_ylim(*self._limits)
        self._axis3D.set_zlim(*self._limits)
        self.progress_slider.setValue(0)
        self._figure3D.canvas.draw()
        self._figureEMG.canvas.draw()

    def _options_pressed(self):
        """
        method handling the options button press events.
        """
        self._stop_player()
        self.options_pane.setVisible(self.option_button.isChecked())

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
        self.speed_box.setValue(self.speed_slider.value())

    def _adjust_markers(self):
        """
        adjust the markers appearance.
        """
        for n in self._Marker3D:
            self._Marker3D[n].set_color(self.marker_options.color())
            self._Marker3D[n].set_ms(self.marker_options.size())
            self._Marker3D[n].zorder = self.marker_options.zorder()
        self._update_figure()

    def _adjust_forces(self):
        """
        adjust the forces appearance.
        """
        for n in self._ForcePlatform3D:
            self._ForcePlatform3D[n].set_color(self.force_options.color())
            self._ForcePlatform3D[n].set_linewidth(self.force_options.size())
            self._ForcePlatform3D[n].zorder = self.force_options.zorder()
        self._update_figure()

    def _adjust_links(self):
        """
        adjust the links appearance.
        """
        for n in self._Link3D:
            self._Link3D[n].set_color(self.link_options.color())
            self._Link3D[n].set_linewidth(self.link_options.size())
            self._Link3D[n].zorder = self.link_options.zorder()
        self._update_figure()

    def _adjust_references(self):
        """
        adjust the reference frame appearance.
        """
        col = self.reference_options.color()
        val = self.reference_options.size()
        order = self.reference_options.zorder()
        for n in self._ReferenceFrame3D:
            self._ReferenceFrame3D[n]["Text"].set_color(col)
            self._ReferenceFrame3D[n]["Text"].zorder = order
            self._ReferenceFrame3D[n]["Versor"].set_color(col)
            self._ReferenceFrame3D[n]["Versor"].set_linewidth(val)
            self._ReferenceFrame3D[n]["Versor"].zorder = order
            if not self.reference_button.isChecked():
                self._ReferenceFrame3D[n]["Text"].set_alpha(0)
                self._ReferenceFrame3D[n]["Versor"].set_alpha(0)
        self._update_figure()

    def _adjust_labels(self):
        """
        adjust the text appearance.
        """
        val = self.text_options.size()
        col = self.text_options.color()
        order = self.text_options.zorder()

        # adjust labels
        for n in self._Text3D:
            self._Text3D[n].set_size(val)
            self._Text3D[n].set_color(col)
            self._Text3D[n].zorder = order
            markers = list(self._Marker3D.keys())
            forces = list(self._ForcePlatform3D.keys())
            if n in markers and not self.marker_button.isChecked():
                self._Text3D[n].set_alpha(0)
            elif n in forces and not self.force_button.isChecked():
                self._Text3D[n].set_alpha(0)

        # adjust reference frame axis labels
        for n in self._ReferenceFrame3D:
            self._ReferenceFrame3D[n]["Text"].set_size(val)

        # update
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
