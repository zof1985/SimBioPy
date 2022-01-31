# KINEMATICS MODULE


#! IMPORTS

from .geometry import *
import numpy as np
import pandas as pd
import plotly.express as px


#! CLASSES


class Marker3D(Vector):
    """
    generate a 3D object reflecting a marker collected over time in a 3D space.

    Parameters
    ----------
    amplitude_data: Point, pandas.DataFrame, numpy.ndarray, list
        the amplitude of the vector.

    index: arraylike
        the index for both the amplitude and origin of the vector.

    columns: arraylike
        the name of the dimensions of the vector's origin and amplitude.
    """

    def __init__(
        self,
        amplitude_data: Tuple[Point, pd.DataFrame, np.ndarray, list],
        index: Tuple[list, np.ndarray] = None,
    ):
        amp = self._get_data(amplitude_data, index, ["X", "Y", "Z"])
        assert amp.ndim == 3, "a 3D dataset must be provided."
        super(self, Marker3D).__init__(amplitude_data=amp)


class Force3D(Vector):
    """
    generate a 3D object reflecting a marker collected over time in a 3D space.

    Parameters
    ----------
    amplitude_data: Point, pandas.DataFrame, numpy.ndarray, list
        the amplitude of the vector.

    origin_data: Point, pandas.DataFrame, numpy.ndarray, list
        the origin of the vector.

    index: arraylike
        the index for both the amplitude and origin of the vector.

    columns: arraylike
        the name of the dimensions of the vector's origin and amplitude.
    """

    def __init__(
        self,
        amplitude_data: Tuple[Point, pd.DataFrame, np.ndarray, list],
        origin_data: Tuple[Point, pd.DataFrame, np.ndarray, list] = None,
        index: Tuple[list, np.ndarray] = None,
    ):
        amp = self._get_data(amplitude_data, index, ["X", "Y", "Z"])
        assert amp.ndim == 3, "a 3D dataset must be provided."
        ori = self._get_data(origin_data, index, ["X", "Y", "Z"])
        assert ori.ndim == 3, "a 3D dataset must be provided."
        super(self, Force3D).__init__(amplitude_data=amp, origin_data=ori)


class Link3D(Segment):
    """
    Generate an object reflecting a dimension-less vector in a n-dimensional space.

    Parameters
    ----------

    p0_data: Point, pandas.DataFrame, numpy.ndarray, list
        the first point data of the segment.

    p1_data: Point, pandas.DataFrame, numpy.ndarray, list
        the second of the segment.

    index: arraylike
        the index for both the amplitude and origin of the segment.
    """

    def __init__(
        self,
        p0_data: Tuple[Point, pd.DataFrame, np.ndarray, list],
        p1_data: Tuple[Point, pd.DataFrame, np.ndarray, list],
        index: Tuple[list, np.ndarray] = None,
    ):
        # get p0 and p1 as Point instances.
        p0 = self._get_data(p0_data, index, ["X", "Y", "Z"])
        assert p0.ndim == 3, "a 3D dataset must be provided."
        p1 = self._get_data(p1_data, index, ["X", "Y", "Z"])
        assert p1.ndim == 3, "a 3D dataset must be provided."
        super(self, Link3D).__init__(p0_data=p0, p1_data=p1)


class KinematicModel3D:
    """
    Generate a 3D model which may include:
        - kinematic markers
        - force platforms data
        - Surface Electromyography data

    Parameters
    ----------
    markers: dict
        A dict containing the markers included in the model.

    forces: dict
        A dict containing the forces included in the model.

    emgs: dict
        A dict containing the EMG channels included in the model.

    links: dict
        A list of segments defining the links between markers.

    frames: dict
        A list of reference frames defining the reference system of the data.
    """

    def __init__(
        self,
        markers: dict = {},
        forces: dict = {},
        emgs: dict = {},
        links: dict = {},
        frames: dict = {},
    ):

        # add the markers
        for i, v in markers.items():
            assert isinstance(v, Point), "'{}' must be a Point instance.".format(i)
        self.markers = markers.copy()

        # add the forces
        for i, v in forces.items():
            assert isinstance(v, Vector), "'{}' must be a Vector instance.".format(i)
        self.forces = forces.copy()

        # add the links
        self.links = pd.DataFrame()
        for i, v in links.items():
            assert isinstance(v, Segment), "'{}' must be a Segment instance.".format(i)
        self.links = links.copy()

        # add the emg channels
        for i, v in emgs.items():
            assert isinstance(v, Point), "'{}' must be a Point instance.".format(i)
        self.emgs = emgs.copy()

        # add the reference frames
        for i, v in frames.items():
            txt = "'{}' must be a ReferenceFrame instance.".format(i)
            assert isinstance(v, ReferenceFrame), txt
        self.frames = frames.copy()

    def stack(self) -> pd.DataFrame:
        """
        stack the provided obj.

        Returns
        -------
        df: pd.DataFrame
            the data stacked into a convenient format.
        """
        out = []

        # markers
        for i, v in self.markers.items():
            new = v.stack()
            new.insert(0, "Label", np.tile(i, new.shape[0]))
            new.insert(0, "Type", np.tile("Marker", new.shape[0]))
            out += [new]

        # forces
        for i, v in self.forces.items():
            new = v.stack()
            new.insert(0, "Label", np.tile(i, new.shape[0]))
            new.insert(0, "Type", np.tile("Force", new.shape[0]))
            out += [new]

        # links
        for i, v in self.links.items():
            new = v.stack()
            new.insert(0, "Label", np.tile(i, new.shape[0]))
            new.insert(0, "Type", np.tile("Link", new.shape[0]))
            out += [new]

        # emg channels
        for i, v in self.emgs.items():
            new = v.stack()
            new.insert(0, "Label", np.tile(i, new.shape[0]))
            new.insert(0, "Type", np.tile("EMG", new.shape[0]))
            out += [new]

        # reference frames
        for i, v in self.frames.items():
            new = v.stack()
            new.insert(0, "Label", np.tile(i, new.shape[0]))
            new.insert(0, "Type", np.tile("ReferenceFrame", new.shape[0]))
            out += [new]

        return pd.concat(out, axis=0, ignore_index=True)
