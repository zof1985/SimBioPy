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


class Model3DWidget(qtw.QWidget):
    """
    renderer for a 3D Model.
    """

    # class objects
    dpi = 300
    data = None
    slider = None
    time_label = None
    play_button = None
    forward_button = None
    backward_button = None
    repeat_button = None
    canvasEMG = None
    navigationBarEMG = None
    canvas3D = None
    navigationBar3D = None
    labels_checkbox = None
    reference_checkbox = None

    # private variables
    _times = None
    _font_size = 12
    _button_size = 35
    _play_timer = None
    _update_rate = 1  # msec
    _is_running = False
    _figure3D = None
    _figureEMG = None
    _axis3D = None
    _axisEMG = None
    _Marker3D = {}
    _ForcePlatform3D = {}
    _Link3D = {}
    _Text3D = {}
    _EmgSensor = {}
    _force3D_sclr = 1
    _actual_frame = None
    _FigureAnimator3D = None
    _FigureAnimatorEMG = None
    _play_start_time = None
    _reference_frame = None
    _path = None

    def __init__(self, model, parent=None):
        """
        constructor
        """
        super(Model3DWidget, self).__init__(parent=parent)

        # check the model
        txt = "model must be a Model3D instance."
        assert isinstance(model, Model3D), txt

        # path to the package folder
        self._path = os.path.sep.join([os.getcwd(), "simbiopy"])

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
                    pp.columns = pd.Index(["{}{}".format(i, l[1]) for i in cols])
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
        self.slider.setMaximum(len(self._times) - 1)
        self.slider.setTickInterval(1)
        self.slider.valueChanged.connect(self._slider_moved)
        self.slider.setFixedHeight(self._button_size)

        # set the timer for the player
        self._play_timer = qtc.QTimer()
        self._play_timer.timeout.connect(self._player)

        # time label
        self.time_label = qtw.QLabel("00:00.000")
        self.time_label.setFont(qtg.QFont("Arial", self._font_size))
        self.time_label.setFixedHeight(self._button_size)
        self.time_label.setFixedWidth(110)
        self.time_label.setAlignment(qtc.Qt.AlignCenter)

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
        commands_widget.setFixedHeight(self._button_size * 1.5)

        # make the EMG pane
        if model.has_EmgSensor():

            # figure
            rows = len(self.data["EmgSensor"])
            grid = GridSpec(rows, 1)
            self._figureEMG = pl.figure(dpi=self.dpi)
            self._figureEMG.tight_layout()
            self.canvasEMG = FigureCanvasQTAgg(self._figureEMG)
            self.canvasEMG.setAutoFillBackground(True)

            # add the emg data
            self._EmgSensor = {}
            emg_axes = []
            for i, s in enumerate(model.EmgSensor):

                # plot the whole EMG signal
                ax = self._figureEMG.add_subplot(grid[rows - 1 - i])
                obj = model.EmgSensor[s].amplitude
                obj = obj.dropna()
                time = obj.index.to_numpy()
                amplitude = obj.values.flatten()
                ax.plot(time, amplitude, linewidth=0.5)

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

                # set the y-axis limits and bounds
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
                self._EmgSensor[s] = ax.plot(
                    x_line,
                    y_line,
                    "--",
                    linewidth=0.4,
                    animated=True,
                )[0]

            # setup the figure animator object
            self._FigureAnimatorEMG = FigureAnimator(
                figure=self._figureEMG,
                artists=[i for i in self._EmgSensor.values()],
            )

        else:

            # generate an empty widget
            self.canvasEMG = qtw.QWidget()

        # create the 3D model pane
        if model.has_ForcePlatform3D() or model.has_Marker3D():

            # generate the axis
            self._figure3D = pl.figure(dpi=self.dpi)
            self._figure3D.tight_layout()
            self.canvas3D = FigureCanvasQTAgg(self._figure3D)
            self.canvas3D.setAutoFillBackground(True)
            self._axis3D = self._figure3D.add_subplot(projection="3d")

            # set the pane view
            self._axis3D.view_init(elev=10, azim=45)

            # make the panes transparent
            self._axis3D.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
            self._axis3D.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
            self._axis3D.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))

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
            maxc = max(0, np.nanmax(edges))
            minc = min(0, np.nanmin(edges))
            self._axis3D.set_xlim(minc, maxc)
            self._axis3D.set_ylim(minc, maxc)
            self._axis3D.set_zlim(minc, maxc)

            # generate the reference frame
            self._reference_frame = [
                self._axis3D.quiver(
                    np.array([0, 0, 0]),
                    np.array([0, 0, 0]),
                    np.array([0, 0, 0]),
                    np.array([0.2, 0, 0]) * (maxc - minc),
                    np.array([0, 0.2, 0]) * (maxc - minc),
                    np.array([0, 0, 0.2]) * (maxc - minc),
                    color="gold",
                    animated=True,
                    length=1,
                    linewidth=0.5,
                ),
                self._axis3D.text(
                    (maxc - minc) * 0.1,
                    0,
                    0,
                    "X",
                    size=3,
                    color="gold",
                    animated=True,
                ),
                self._axis3D.text(
                    0,
                    (maxc - minc) * 0.1,
                    0,
                    "Y",
                    size=3,
                    color="gold",
                    animated=True,
                ),
                self._axis3D.text(
                    0,
                    0,
                    (maxc - minc) * 0.1,
                    "Z",
                    size=3,
                    color="gold",
                    animated=True,
                ),
            ]

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
                            l,
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
                            l,
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
            artists = [i for i in self._reference_frame]
            artists += [i for i in self._Marker3D.values()]
            artists += [i for i in self._ForcePlatform3D.values()]
            artists += [i for i in self._Text3D.values()]
            artists += [i for i in self._Link3D.values()]
            self._FigureAnimator3D = FigureAnimator(
                figure=self._figure3D,
                artists=artists,
            )

        else:

            # generate an empty widget
            self.canvas3D = qtw.QWidget()

        # 3D pane
        pane_3d = qtw.QWidget()
        if self._figure3D is not None:

            self.labels_checkbox = qtw.QCheckBox()
            self.labels_checkbox.setChecked(True)
            self.labels_checkbox.setText("Print Labels")
            self.labels_checkbox.stateChanged.connect(self._update_figure)

            self.reference_checkbox = qtw.QCheckBox()
            self.reference_checkbox.setChecked(True)
            self.reference_checkbox.setText("Show reference")
            self.reference_checkbox.stateChanged.connect(self._update_reference)

            top_bar = qtw.QWidget()
            top_bar_layout = qtw.QHBoxLayout()
            top_bar_layout.addWidget(self.reference_checkbox)
            top_bar_layout.addWidget(self.labels_checkbox)
            top_bar.setLayout(top_bar_layout)

            pane_3d_layout = qtw.QVBoxLayout()
            pane_3d_layout.addWidget(top_bar)
            pane_3d_layout.addWidget(self.canvas3D)
            pane_3d.setLayout(pane_3d_layout)

        # image pane
        splitter = qtw.QSplitter(qtc.Qt.Horizontal)
        splitter.addWidget(pane_3d)
        splitter.addWidget(self.canvasEMG)

        # widget layout
        layout = qtw.QVBoxLayout()
        layout.addWidget(splitter)
        layout.addWidget(commands_widget)
        self.setLayout(layout)

        # set the actual frame
        self._actual_frame = 0

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
        button.setFixedHeight(self._button_size)
        button.setFixedWidth(self._button_size)
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
        self._play_start_time = get_time()
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

    def _player(self):
        """
        player event handler
        """
        lapsed = (get_time() - self._play_start_time) * 1000 + self._times[0]
        if lapsed > self._times[-1] - self._times[0]:
            self._play_start_time = get_time()
            self.slider.setValue(0)
        else:
            self.slider.setValue(np.argmin(abs(self._times - lapsed)))

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
                    self._ForcePlatform3D[t]._alpha = s
                    self._Text3D[t]._x = x
                    self._Text3D[t]._y = y
                    self._Text3D[t]._z = z
                    if self.labels_checkbox.isChecked():
                        self._Text3D[t]._alpha = 1
                    else:
                        self._Text3D[t]._alpha = 0

                elif sns == "Link3D":
                    x, y, z, u, v, w, s = df.loc[time].values
                    self._Link3D[t].set_data_3d(
                        np.array([x, u]),
                        np.array([y, v]),
                        np.array([z, w]),
                    )
                    self._Link3D[t]._alpha = s

                elif sns == "Marker3D":
                    x, y, z, s = df.loc[time].values
                    self._Marker3D[t].set_data_3d(x, y, z)
                    self._Marker3D[t]._alpha = s
                    self._Text3D[t]._x = x
                    self._Text3D[t]._y = y
                    self._Text3D[t]._z = z
                    if self.labels_checkbox.isChecked():
                        self._Text3D[t]._alpha = 1
                    else:
                        self._Text3D[t]._alpha = 0

                elif sns == "EmgSensor":
                    x, y, s = df.loc[time].values
                    self._EmgSensor[t].set_xdata([x, y])

        # update the figures
        if self._figure3D is not None:
            self._FigureAnimator3D.update()

        if self._figureEMG is not None:
            self._FigureAnimatorEMG.update()

    def _update_reference(self):
        """
        update the reference frame visibility.
        """
        # adjust the reference frame
        for i in range(len(self._reference_frame)):
            if self.reference_checkbox.isChecked():
                self._reference_frame[i].set_color("gold")
            else:
                self._reference_frame[i].set_color((1, 1, 1, 0))
        self._update_figure()
