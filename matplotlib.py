# LABELS MODULE


#! IMPORTS

import abc
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as pl
from matplotlib.backend_bases import MouseEvent
from matplotlib import colors, axis, patches, transforms


#! CLASSES


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

    def __init__(self, figure, artists:mpl.axis.):
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


class Label(metaclass=abc.ABCMeta):
    """
    generic implementation of Label.
    This class is designed to deal with all the calculations required to
    draw a label on a matplotlib figure using just the mouse.

    To use this class:
    -   The user must implement this class providing the desired patches.Patch
        as input.
    -   Implement the "update_patch" method with form:

            def update_patch(self) -> None:
                --- your code here ---

        This method is used to tell the Label instance how to update the patch
        according to the mouse position. This is automatically handled by the
        base Label class. Therefore there is no need to pass neither the
        coordinates of the mouse, not the event triggered the update.

    Parameters
    ----------
    name: str
        the name of the Label

    axes: List[matplotlib.axis.Axis] | matplotlib.axis.Axis
        the list of axes on which the label has to be rendered.

    patch: matplotlib.patches.Patch
        the type of patch defining the label.

    color: str | tuple = (0, 0, 0, 0)
        the color of the patch.
    """

    # ****** VARIABLES ****** #

    _name = ""
    _line = None
    _patch = None
    _text = None
    _mouse_p1 = None
    _mouse_p0 = None
    _mouse_pressed = False
    _drawn = False
    _axes = []
    _lines = []
    _patches = []

    # ****** CONSTRUCTOR ****** #

    def __init__(
        self,
        name: str,
        axes: list | axis.Axis,
        patch: patches.Patch,
    ) -> None:
        self.set_name(name)
        self.set_axes(axes)
        self.set_patch(patch)
        self.set_line()
        self.set_text()
        for axs in self.axes:
            self._patches += [axs.add_artist(self._patch)]
            self._lines += [axs.add_artist(self._line)]
            self._texts += [axs.add_artist(self._text)]
        self.reset()

    # ****** SETTERS ****** #

    def set_name(self, name: str) -> None:
        """Store the name of the label"""
        if not isinstance(name, str):
            raise TypeError("name must be a str instance.")
        self._name = name

    def set_axes(self, axes: list | axis.Axis) -> None:
        """Store the axes on which the Label has to be rendered"""
        mpl_axes = [axes] if not isinstance(axes, list) else axes
        for axs in mpl_axes:
            if not isinstance(axs, axis.Axis):
                msg = f"{axs} must be a matplotlib.axis.Axis object."
                raise TypeError(msg)
        self._axes = mpl_axes
        for axs in self._axes:
            canvas = axs.figure.canvas
            canvas.mpl_connect("pick_event", self.on_mouse_press)
            canvas.mpl_connect("button_release_event", self.on_mouse_release)
            canvas.mpl_connect("motion_notify_event", self.on_mouse_motion)
            canvas.mpl_connect("scroll_event", self.on_scroll)

    def set_alpha(self, alpha: float) -> None:
        """set the alpha value of the Label"""
        if not isinstance(alpha, float) or alpha < 0 or alpha > 1:
            msg = "alpha must be a float value within the [0, 1] range."
            raise TypeError(msg)
        self._alpha = alpha
        face_color = colors.to_rgba_array(self._patch.facecolor, alpha / 2)
        self._patch.set_facecolor(face_color)
        edge_color = colors.to_rgba_array(self._patch.facecolor, alpha)
        self._patch.set_edgecolor(edge_color)

    def set_color(self, color: str | tuple, alpha: float | None = 1) -> None:
        """set the Label color and alpha."""
        try:
            rgba = colors.to_rgba_array(color, alpha)
            self._color = rgba[:-1]
            self.set_alpha(rgba[-1])
        except Exception as exc:
            msg = f"{color} is not a valid matplotlib color."
            raise TypeError(msg) from exc

    def set_patch(self, patch: patches.Patch) -> None:
        """Store the Label patch"""
        if not isinstance(patch, patches.Patch):
            raise TypeError(f"patch must be a {patches.Patch} instance.")
        self._patch = patch
        self._patch.set_visible(False)
        self._patch.set_antialiased(True)
        self._patch.set_animated(True)
        self._patch.set_linewidth(4)
        self._patch.set_picker(True)

    def set_line(self) -> None:
        """setup the line connection used whilst drawing the label."""
        line_options = dict(coordsA="data", coordsB="data", arrowstyle="-")
        line_options.update(**dict(linewidth=4, linestyle="dashed", shrinkA=5))
        line_options.update(**dict(shrinkB=5, mutation_scale=20, visible=False))
        line_options.update(**dict(alpha=0.5, facecolor="k", edgecolor="k"))
        line_options.update(**dict(antialiased=True, animated=True))
        self._line = patches.ConnectionPatch([0, 0], [0, 0], **line_options)

    def set_text(self) -> None:
        """setup the text linked to the label."""
        self._text = pl.text(
            x=0,
            y=0,
            s=self._name,
            c=self._patch.facecolor,
            a=self._patch.alpha,
            animated=True,
        )

    # ****** GETTERS ****** #

    @property
    def name(self) -> str:
        """return the name of the label"""
        return self._name

    @property
    def axes(self) -> str:
        """return the axes on which the Label has to be rendered"""
        return self._axes

    @property
    def color(self) -> tuple:
        """return the Label face and edge colors"""
        return self._color

    @property
    def alpha(self) -> float:
        """return the Label alpha"""
        return self._alpha

    @property
    def patch(self) -> str:
        """return the patch of the label"""
        return self._patch

    @property
    def artists(self) -> list:
        """return the list of artists used to render the Label"""
        return self._patches + self._lines

    # ****** EVENTS HANDLERS ****** #

    def on_mouse_press(self, event: MouseEvent) -> None:
        """mouse press event"""
        if event.button == 1:
            self._mouse_pressed = True
            if event.dblclick:  # if the Label is double-clicked delete it
                self.reset()
            elif not self.is_drawn():  # update mouse click position
                self._update_mouse_p0(event)

    def on_mouse_motion(self, event: MouseEvent) -> None:
        """mouse motion event"""
        if self.is_mouse_pressed():
            self._update_mouse_p1(event)
            if self.is_drawn():  # translate the patch
                self._move()
            else:  # draw the patch and the line
                self._update_line()
                self.update_patch()
            self.update_view()  # update the view

    def on_mouse_release(self, event: MouseEvent) -> None:
        """mouse release event"""
        if self._mouse_p0 is not None and self._mouse_p1 is not None:
            self.update_patch()
            self._drawn = True
        self._mouse_pressed = False
        self._mouse_p1 = None
        self._update_line()
        self.update_view()

    def on_scroll(self, event: MouseEvent) -> None:
        """scrolling action (rotates the artists)"""
        if self.is_drawn():
            angle = 10 / 180 * np.pi * event.step
            trans = transforms.Affine2D().rotate_deg(angle)
            self._apply_transformation(trans)

    # ****** METHODS ****** #

    def is_mouse_pressed(self) -> bool:
        """return whether the mouse button is pressed"""
        return self._mouse_pressed

    def is_drawn(self) -> bool:
        """check if the actual patch has been drawn on screen."""
        return self._drawn

    def reset(self) -> None:
        """remove the current Label from being seen."""
        self._patch.set_visible(False)
        self._line.set_visible(False)
        self._mouse_p0 = None
        self._mouse_p1 = None

    def _apply_transformation(self, trans: transforms.Affine2D) -> None:
        """apply the given transformation to all artists"""
        if not isinstance(trans, transforms.Affine2D):
            msg = f"transformation must be a {transforms.Affine2D} instance."
            raise TypeError(msg)

    def _move(self) -> None:
        """translate the actual patch according to the mouse motion"""
        if self._mouse_p1 is not None:
            trans = transforms.Affine2D().translate(*self._mouse_p1)
            self._apply_transformation(trans)

    def _update_mouse_p0(self, event: MouseEvent) -> None:
        """set the mouse motion origin."""
        self._mouse_p0 = np.array([event.xdata, event.ydata])

    def _update_mouse_p1(self, event: MouseEvent) -> None:
        """set the mouse motion end."""
        self._mouse_p1 = np.array([event.xdata, event.ydata])

    def update_view(self) -> None:
        """update the view of the object."""
        for artist in self.artists:
            artist.figure.canvas.draw()

    def _update_line(self) -> None:
        """update the line settings."""
        self._line.set_visible(self.is_mouse_pressed())
        if self._mouse_p0 is not None and self._mouse_p1 is not None:
            self._line.set_positions(self._mouse_p0, self._mouse_p1)

    @abc.abstractmethod
    def update_patch(self) -> None:
        """update the patch settings."""
        raise NotImplementedError

    # ****** CLASS METHODS ****** #

    @classmethod
    def __subclasshook__(cls, subclass) -> NotImplemented | bool:
        return hasattr(subclass, "update_patch") or NotImplemented


class Rectangle(Label):
    """
    make a rectangle label.

    Parameters
    ----------
    name: str
        the name of the Label

    axes: List[matplotlib.axis.Axis] | matplotlib.axis.Axis
        the list of axes on which the label has to be rendered.

    color: str | tuple = (0, 0, 0, 0)
        the color of the patch.
    """

    def __init__(
        self,
        name: str,
        axes: list | axis.Axis,
        color: str | tuple = (0, 0, 0, 0),
    ) -> None:
        super().__init__(
            name=name,
            axes=axes,
            patch=patches.Rectangle((0, 0), 1, 1),
            color=color,
        )

    def update_patch(self) -> None:
        """update the rectangle label patch"""
        delta = self._mouse_p1 - self._mouse_p0
        self._patch.set_width(delta[0])
        self._patch.set_height(delta[1])
        self._patch.set_xy(self._mouse_p0)
        for artist in self._patches:
            artist.set_width(delta[0])
            artist.set_height(delta[1])
            artist.set_xy(self._mouse_p0)


class Rectangle(Label):
    """
    make a rectangle label.

    Parameters
    ----------
    name: str
        the name of the Label

    axes: List[matplotlib.axis.Axis] | matplotlib.axis.Axis
        the list of axes on which the label has to be rendered.

    color: str | tuple = (0, 0, 0, 0)
        the color of the patch.
    """

    def __init__(
        self,
        name: str,
        axes: list | axis.Axis,
        color: str | tuple = (0, 0, 0, 0),
    ) -> None:
        super().__init__(
            name=name,
            axes=axes,
            patch=patches.Rectangle((0, 0), 1, 1),
            color=color,
        )

    def update_patch(self) -> None:
        """update the rectangle label patch"""
        delta = self._mouse_p1 - self._mouse_p0
        self._patch.set_width(delta[0])
        self._patch.set_height(delta[1])
        self._patch.set_xy(self._mouse_p0)
        for artist in self._patches:
            artist.set_width(delta[0])
            artist.set_height(delta[1])
            artist.set_xy(self._mouse_p0)
