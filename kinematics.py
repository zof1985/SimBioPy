# IMPORTS


from io import BufferedReader
from .base import *
from typing import Tuple
import os
import struct
import weakref
import numpy as np
import pandas as pd
import matplotlib.pyplot as pl


#! CLASSES


class ReferenceFrame:
    """
    Create a ReferenceFrame instance.

    Attributes
    ----------

    origin: array-like 1D
        a 1D list that contains the coordinates of the ReferenceFrame's origin.

    orientation: NxN array
        a 2D square matrix containing on each row the versor corresponding to each
        dimension of the origin.

    names: array-like 1D
        a 1D array with the names of the dimensions.
    """

    def __init__(self, origin, orientation=None, names=None):
        """
        Create a ReferenceFrame object.

        Parameters
        ----------

        origin: array-like 1D
            a 1D list that contains the coordinates of the ReferenceFrame's origin.

        orientation: NxN array
            a 2D square matrix containing on each row the versor corresponding to each
            dimension of the origin.

        names: array-like 1D
            a 1D array with the names of the dimensions.
        """

        def check_input(inp, lbl):
            txt = "{} must be a list or numpy array."
            assert isinstance(inp, (list, np.ndarray)), txt.format(lbl)

        # handle the origin parameter
        check_input(origin, "origin")
        self.origin = np.atleast_1d(origin).flatten()

        # handle the names parameter
        if names is None:
            names = ["X{}".format(i + 1) for i in range(len(self.origin))]
        check_input(names, "names")
        self.names = np.squeeze(names).flatten()

        # handle the orientation parameter
        if orientation is None:
            orientation = np.eye(len(self.origin))
        check_input(orientation, "orientation")
        self.orientation = self._gram_schmidt(orientation)

    def __str__(self):
        """
        generate a pandas.DataFrame representing the ReferenceFrame.
        """
        origin_df = pd.DataFrame(
            data=np.atleast_2d(self.origin), columns=self.names, index=["Origin"]
        )
        orientation_df = pd.DataFrame(
            data=self.orientation,
            columns=self.names,
            index=self.names,
        )
        return origin_df.append(orientation_df).__str__()

    def __repr__(self):
        return self.__str__()

    def _matches(self, vector):
        """
        Check vector may be part of self.

        Parameters
        ----------
        vector: Vector, Vector
            a Vector or Vector object.

        Returns
        -------
        matches: bool
            True if vector can be part of self. False, otherwise.
        """

        if not isinstance(vector, pd.DataFrame):
            return False
        if len(self.origin) != vector.shape[1]:
            return False
        if not np.all([np.any(vector.columns == i) for i in self.names]):
            return False
        return True

    def copy(self):
        """
        Return a copy of self.
        """
        return ReferenceFrame(
            origin=self.origin, orientation=self.orientation, names=self.names
        )

    def rotate(self, vector):
        """
        Rotate vector according to the current ReferenceFrame instance.

        Parameters
        ----------
        vector: Vector
            a Vector object.

        Returns
        -------
        vec: Vector
            a rotated Vector object.
        """
        assert self._matches(vector), "vector must be a Vector object."
        ov = np.vstack([np.atleast_2d(self.origin) for _ in range(vector.shape[0])])
        out = (vector - ov).dot(self.orientation.T)
        out.columns = self.names
        return out

    def invert(self, vector):
        """
        Rotate vector back from the current Reference Frame to its former one.

        Parameters
        ----------
        vector: Vector
            a Vector object.

        Returns
        -------
        vec: Vector
            a rotated Vector object.
        """
        assert self._matches(vector), "vector must be a Vector object."
        ov = np.vstack([np.atleast_2d(self.origin) for _ in range(vector.shape[0])])
        out = vector.dot(self.orientation)
        out.columns = self.names
        return out + ov

    def _gram_schmidt(self, vectors):
        """
        Return the orthogonal basis defined by a set of vectors using the
        Gram-Schmidt algorithm.

        Parameters:
            vectors (np.ndarray): a NxN numpy.ndarray to be orthogonalized (by row).

        Returns:
            a NxN numpy.ndarray containing the orthogonalized arrays.
        """

        def proj(a, b):
            return (np.inner(a, b) / np.inner(b, b) * b).astype(float)

        def norm(v):
            return v / np.sqrt(np.sum(v ** 2))

        # calculate the projection points
        W = []
        for i, u in enumerate(vectors):
            w = np.copy(u).astype(float)
            for j in vectors[:i, :]:
                w -= proj(u, j)
            W += [w]

        # normalize
        return np.vstack([norm(u) for u in W])


class Vector(pd.DataFrame):

    # attributes
    _cacher = ()
    dtype = np.float64

    def __init__(self, *args, **kwargs):
        if any([i == "name" for i in kwargs]):
            args = [pd.Series(*args, **kwargs)]
            kwargs = {}
        super(Vector, self).__init__(*args, **kwargs)

    def _set_as_cached(self, item, cacher) -> None:
        """
        Set the _cacher attribute on the calling object with a weakref to
        cacher.
        """
        self._cacher = (item, weakref.ref(cacher))

    def __repr__(self) -> str:
        return pd.DataFrame(self).__repr__()

    def __str__(self) -> str:
        return pd.DataFrame(self).__str__()

    @property
    def _constructor(self):
        return Vector

    @property
    def _constructor_sliced(self):
        return Vector

    def dropna(self, axis: int = 0):
        """
        return the Vector without missing data.

        Parameters
        ----------
        axis: int
            the dimension along with dropna must work.

        Returns
        -------
        full: Vector
            the vector without missing data.
        """
        return Vector(pd.DataFrame(self).dropna(axis=axis, inplace=False))

    def unique(self, axis: int = 0):
        """
        return the unique rows or columns in the vector.

        Parameters
        ----------
        axis: int
            the dimension along with unique must work.

        Returns
        -------
        full: Vector
            the vector unique occurrences.
        """
        val, ix = np.unique(self.values, return_index=True, axis=axis)
        if axis == 0:
            idx = self.index.to_numpy()[ix]
            col = self.columns.to_numpy()
        elif axis == 1:
            idx = self.index.to_numpy()
            col = self.columns.to_numpy()[ix]
        else:
            raise ValueError("'axis' must be 0 or 1.")
        return Vector(val, index=idx, columns=col)

    def copy(self):
        """
        return a copy of the current vector.
        """
        return Vector(self)

    @property
    def T(self):
        """
        return the transpose of the vector
        """
        return Vector(pd.DataFrame(self).T)

    def plot(self, show=True, *args, **kwargs):
        """
        generate a matplotlib plot representing the current object.

        Parameters
        ----------
        input parameters as described here:
        https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.plot.html

        Returns
        -------
        a matplotlib.Figure object.
        """
        if not show:
            return pd.DataFrame(self).plot(*args, **kwargs)
        else:
            pd.DataFrame(self).plot(*args, **kwargs)
            pl.show()

    def sampling_frequency(self, digits=3):
        """
        return the "average" sampling frequency of the Vector.

        Parameters
        ----------
        digits: int
            the number of digits for the returing value
        """
        return np.round(1.0 / np.mean(np.diff(self.index.to_numpy())), digits)

    @property
    def norm(self):
        """
        get the norm of the vector.
        """
        dt = np.sqrt(np.sum(self.values ** 2, axis=1))
        lbls = [str(i) for i in self.columns.to_list()]
        cols = "|{}|".format("+".join(lbls))
        idx = self.index
        return Vector(data=dt, columns=[cols], index=idx)

    def get_angle(self, x: str, y: str, name: str = None):
        """
        return the angle between dimensions y and x using the arctan function.

        Parameters
        ----------
        x, y: str
            the name of the columns to be used for the angle calculation.

        name: str or None
            the name of the output dataframe column

        Returns
        -------
        q: Vector
            the angle in radiants.
        """

        # check the data
        if name is None:
            name = "Angle"
        assert isinstance(name, str), "name must be a string"
        assert isinstance(x, str), "x must be a string"
        assert x in self.columns.to_list(), "x not in columns."
        assert isinstance(y, str), "y must be a string"
        assert y in self.columns.to_list(), "y not in columns."

        # calculate the angle
        q = np.arctan2(self[y].values.flatten(), self[x].values.flatten())
        return Vector(q, columns=[name], index=self.index)

    def matches(self, vector: pd.DataFrame) -> bool:
        """
        check if the vector V is comparable to self.

        Parameters
        ----------
        vector:  Vector
            the vector to be compared

        Returns
        -------
        Q: bool
            true if V matches self, False otherwise.
        """
        # check the shape
        s_rows, s_cols = self.shape
        v_rows, v_cols = vector.shape
        if s_rows != v_rows or s_cols != v_cols:
            return False

        # check the column names
        s_cols = self.columns.to_numpy()
        v_cols = vector.columns.to_numpy()
        if not all([i == j for i, j in zip(s_cols, v_cols)]):
            return False

        # check the index values
        s_idx = self.index.to_numpy()
        v_idx = vector.index.to_numpy()
        if not all([i == j for i, j in zip(s_idx, v_idx)]):
            return False

        return True

    def fillna(self, value: float = None, n_predictors: int = 3, predictors: list = []):
        """
        fill missing values in the vector.

        Parameters
        ----------
        value: float or None
            the value to be used for missing data replacement.
            if None, cubic spline interpolation is used to extract the
            missing data.
            Please note that if predictors is not empty, this parameter is ignored.

        n_predictors: int
            the number of predictors to be used if predictors is not empty

        predictors: list
            list of Vectors that can be matched with self and that can be used to obtain
            missing coordinates via multiple linear regression.
            If left empty, cubic spline interpolation or constant value substitution is
            used according to the inputs provided in value.

        Returns
        -------
        filled: Vector
            the vector without missing data.
        """
        # check if missing values exist
        miss = self.isna().any(1).values.flatten()
        filled = self.copy()

        # otherwise return a copy of the actual vector
        if not np.any(miss):
            return filled

        # get the vector index
        x_new = filled.index.to_numpy()

        # multiple linear regression
        assert isinstance(predictors, list), "'predictors' must be a 'list'."
        if len(predictors) > 0:
            assert isinstance(n_predictors, int), "'n_predictors' must be an 'int'."

            def corr(a, b):
                """
                get the mean correlation between a and b
                """
                cvec = []
                for d in a.columns:
                    v = [a[d].values.flatten(), b[d].values.flatten()]
                    v = np.vstack(np.atleast_2d(v))
                    valid = v[:, np.all(~np.isnan(v), axis=0)]
                    if valid.shape[1] > 0:
                        cvec += [abs(np.corrcoef(valid, rowvar=True)[0, 1])]
                    else:
                        cvec += [0]
                return np.mean(cvec)

            # get mean correlation between self and the predictors
            y = self.values
            corrs = []
            for i in predictors:
                assert self.matches(i), "{} does not match with self.".format(i)
                if np.all(i.loc[miss].notna().values):
                    ix = pd.concat([self, i], axis=1).dropna().index.to_numpy()
                    if len(ix) >= 2:
                        corrs += [corr(self.loc[ix], i.loc[ix])]

            # keep the best n_predictors
            best_preds = np.argsort(corrs)[::-1][:n_predictors]
            best_preds = [v for i, v in enumerate(predictors) if i in best_preds]
            x = pd.concat(best_preds, axis=1).values

            # get the predictive equation
            valid = np.all(~np.isnan(np.concatenate([y, x], axis=1)), axis=1)
            lr = LinearRegression(y[valid], x[valid], True)

            # replace missing values
            filled.loc[miss, filled.columns] = lr.predict(x[miss]).values

        # fill by cubic spline interpolation
        elif value is None:
            df_old = self.dropna()
            for i in filled.columns:
                y_new = interpolate_cs(
                    y=df_old[i].values.flatten(),
                    x_old=df_old.index.to_numpy(),
                    x_new=x_new,
                )
                filled.loc[x_new, [i]] = np.atleast_2d(y_new).T

        # constant value
        else:
            vals = filled.values
            nans = np.argwhere(np.isnan(vals))
            vals[nans] = value
            filled.loc[x_new, filled.columns] = vals

        return filled


#! METHODS


def read_emt(path):
    """
    Create a dict of Vector objects from an excel path.

    Parameters
    ----------
    path: str
        an existing emt path.

    Returns
    -------
    vectors: dict
        a dict with the imported vectors.
    """

    # check the validity of the entered path
    assert os.path.exists(path), path + " does not exist."
    assert path[-4:] == ".emt", path + ' must be an ".emt" path.'

    # read the path
    try:
        path = open(path, "r")
        lines = [[j.strip() for j in i] for i in [i.split("\t") for i in path]]
    except Exception:
        lines = []
    finally:
        path.close()

    # get the output dict
    vd = {}

    # get an array with all the variables
    vrs = np.array([i for i in lines[10] if i != ""]).flatten()

    # get the data names
    names = np.unique([i.split(".")[0] for i in vrs[2:] if len(i) > 0])

    # get the data values
    values = np.vstack([np.atleast_2d(i[: len(vrs)]) for i in lines[11:-2]])
    values = values.astype(float)

    # get the columns of interest
    cols = np.arange(np.argwhere(vrs == "Time").flatten()[0] + 1, len(vrs))

    # get the rows in the data to be extracted
    rows = np.argwhere(np.any(~np.isnan(values[:, cols]), 1)).flatten()
    rows = np.arange(np.min(rows), np.max(rows) + 1)

    # get time
    time = values[rows, 1].flatten()

    # generate a dataframe for each variable
    for v in names:

        # get the dimensions
        D = [i.split(".")[-1] for i in vrs if i.split(".")[0] == v]
        D = [""] if len(D) == 1 else D

        # get the data for each dimension
        nn = []
        coordinates = []
        for i in D:
            nn += [i if i != "" else v]
            cols = np.argwhere(vrs == v + (("." + i) if i != "" else ""))
            coordinates += [values[rows, cols.flatten()]]

        # setup the output variable
        vd[v] = Vector(
            data=np.vstack(np.atleast_2d(coordinates)).T,
            index=time,
            columns=nn,
        )

    return vd


def read_tdf(path: str, fit_to_kinematics: bool = False) -> dict:
    """
    Return the readings from a .tdf file as dicts of Vectors objects.

    Parameters
    ----------
    path: str
        an existing emt path.

    fit_to_kinematics: bool
        should the data be resized according to kinematics readings?
        if True, all data are fit such as the start and end of the
        data match with the start and end of the kinematic data.

    Returns
    -------
    a dict containing the distinct data properly arranged by type.
    """

    # check the validity of the entered path
    assert os.path.exists(path), path + " does not exist."
    assert path[-4:] == ".tdf", path + ' must be an ".tdf" path.'

    # file reader with basic info
    def readFile(file: str, offset: int) -> Tuple[BufferedReader, int, int, int, float]:
        """
        read the file and return it after reading with basic info on it.

        Parameters
        ----------
        file: str
            the path to the file

        offset: int
            the offset to be applied to the file read before extracting the
            relevant data

        Returns
        -------

        fid: BufferedReader
            the file read.

        rows: int
            the number of rows defining the data

        cols: int
            the number of columns defining the data

        freq: int
            the sampling frequency

        time: float
            the time offset of the acquisition
        """
        # read the file
        fid = open(file, "rb")
        fid.seek(offset)

        # get the basic info
        rows, freq, time, cols = struct.unpack("iifi", fid.read(16))
        return fid, rows, cols, freq, time

    def readTracks(
        fid: BufferedReader,
        n_frames: int,
        n_tracks: int,
        freq: int,
        time: float,
        by_frame: bool,
        size: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        internal method used to extract 3D tracks from tdf file.

        Parameters
        ----------
        fid: BufferedReader
            the file read.

        nFrames: int
            the number of samples denoting the tracks.

        nTracks: int
            the number of tracks defining the output.

        freq: int
            the sampling frequency in Hz.

        time: float
            the starting sampling time in s.

        by_frame: bool
            should the data be read by frame or by track?

        size: int
            the expected number of channels for each track
            cameras: 3
            force platforms: 9.

        Returns
        -------
        tracks: numpy.ndarray
            a 2D array with the extracted tracks

        labels: numpy.ndarray
            a 1D array with the labels of each tracks column

        index: numpy.ndarray
            the time index of each track row
        """

        # prepare the arrays for the tracks and the labels
        labels = [u""] * n_tracks
        tracks = np.ones((n_frames, size * n_tracks)) * np.nan

        # read the data
        for trk in range(n_tracks):

            # get the label
            blbl = fid.read(256)
            lbl = ""
            while chr(blbl[0]) != "\x00" and len(blbl) > 0:
                lbl += chr(blbl[0])
                blbl = blbl[1:]
            labels[trk] = lbl

            # read data
            if by_frame:
                n = size * n_tracks * n_frames
                segments = struct.unpack("%if" % n, fid.read(n * 4))
                tracks = np.array(segments).reshape(n_frames, size * n_tracks).T

            # read by track
            else:
                (n_seg,) = struct.unpack("i", fid.read(4))
                fid.seek(4, 1)
                segments = struct.unpack("%ii" % (2 * n_seg), fid.read(8 * n_seg))
                segments = np.array(segments).reshape(n_seg, 2).T
                shape = "{}f".format(size)
                for s in range(n_seg):
                    for row in range(segments[0, s], segments[0, s] + segments[1, s]):
                        val = struct.unpack(shape, fid.read(4 * size))
                        if row < n_frames:
                            cols = np.arange(size * trk, size * trk + size)
                            tracks[row, cols] = val

        # calculate the index
        idx = np.arange(n_frames) / freq + time

        # return the tracks
        fid.close()
        return tracks, labels, idx

    def getPoint3D(file: str, info: dict) -> dict:
        """
        read Point3D tracks data from the provided tdf file.

        Paramters
        ---------
        file: str
            the path to the tdf file

        info: dict
            a dict extracted from the tdf file reading with the info
            required to extract Point3D data from it.

        Returns
        -------
        points: dict
            a dict with all the tracks provided as SimbioPy.Vector objects.
        """
        # get the file read
        fid, n_frames, n_tracks, freq, time = readFile(file, info["Offset"])

        # calibration data (read but not exported)
        _ = np.array(struct.unpack("3f", fid.read(12)))
        _ = np.array(struct.unpack("9f", fid.read(36))).reshape(3, 3).T
        _ = np.array(struct.unpack("3f", fid.read(12)))
        fid.seek(4, 1)

        # check if links exists
        if info["Format"] in [1, 3]:
            (n_links,) = struct.unpack("i", fid.read(4))
            fid.seek(4, 1)
            links = struct.unpack("%ii" % (2 * n_links), fid.read(8 * n_links))
            links = np.array(links)

        # check if the file has to be read by frame or by track
        by_frame = info["Format"] in [3, 4]
        by_track = info["Format"] in [1, 2]
        if not by_frame and not by_track:
            raise IOError("Invalid 'Format' info {}".format(info["Format"]))

        # read the data
        tracks, labels, index = readTracks(
            fid, n_frames, n_tracks, freq, time, by_frame, 3
        )

        # generate the output dict
        points = {}
        for trk in range(n_tracks):
            cols = np.arange(3) + 3 * trk
            points[labels[trk]] = Vector(
                data=tracks[:, cols],
                columns=["X", "Y", "Z"],
                index=index,
            )
        return {"Point3D": points}

    def getForce3D(file: str, info: dict) -> Tuple[dict, dict, dict]:
        """
        read Force3D tracks data from the provided tdf file.

        Paramters
        ---------
        file: str
            the path to the tdf file

        info: dict
            a dict extracted from the tdf file reading with the info
            required to extract Point3D data from it.

        Returns
        -------
        points: dict
            a dict with all the tracks provided as SimbioPy.Vector objects.
        """
        # get the file read (tracks and frames are inverted)
        fid, n_tracks, n_frames, freq, time = readFile(file, info["Offset"])

        # calibration data (read but not exported)
        _ = np.array(struct.unpack("3f", fid.read(12)))
        _ = np.array(struct.unpack("9f", fid.read(36))).reshape(3, 3).T
        _ = np.array(struct.unpack("3f", fid.read(12)))
        fid.seek(4, 1)

        # check if the file has to be read by frame or by track
        by_frame = info["Format"] in [2]
        by_track = info["Format"] in [1]
        if not by_frame and not by_track:
            raise IOError("Invalid 'Format' info {}".format(info["Format"]))

        # read the data
        tracks, labels, index = readTracks(
            fid, n_frames, n_tracks, freq, time, by_frame, 9
        )

        # generate the output dict
        points = {}
        forces = {}
        moments = {}
        for trk in range(n_tracks):
            point_cols = np.arange(3) + 9 * trk
            points[labels[trk]] = Vector(
                data=tracks[:, point_cols],
                columns=["X", "Y", "Z"],
                index=index,
            )
            force_cols = np.arange(3) + 3 + 9 * trk
            forces[labels[trk]] = Vector(
                data=tracks[:, force_cols],
                columns=["X", "Y", "Z"],
                index=index,
            )
            moment_cols = np.arange(3) + 6 + 9 * trk
            moments[labels[trk]] = Vector(
                data=tracks[:, moment_cols],
                columns=["X", "Y", "Z"],
                index=index,
            )
        return {"Point3D": points, "Force3D": forces, "Moment3D": moments}

    def getEMG(file: str, info: str) -> dict:
        """
        read EMG tracks data from the provided tdf file.

        Paramters
        ---------
        file: str
            the path to the tdf file

        info: dict
            a dict extracted from the tdf file reading with the info
            required to extract Point3D data from it.

        Returns
        -------
        channels: dict
            a dict with all the EMG channels provided as SimbioPy.Vector.
        """
        # get the file read (tracks and frames are inverted here)
        fid, n_tracks, n_frames, freq, time = readFile(file, info["Offset"])

        # check if the file has to be read by frame or by track
        by_frame = info["Format"] in [2]
        by_track = info["Format"] in [1]
        if not by_frame and not by_track:
            raise IOError("Invalid 'Format' info {}".format(info["Format"]))

        # read the data
        fid.read(n_tracks * 2)
        tracks, labels, index = readTracks(
            fid, n_frames, n_tracks, freq, time, by_frame, 1
        )

        # generate the output
        channels = Vector(tracks, index=index, columns=labels)
        return {"EMG": {i: channels[[i]] for i in channels}}

    def getIMU(file: str, info: str) -> dict:
        """
        read IMU tracks data from the provided tdf file.

        Paramters
        ---------
        file: str
            the path to the tdf file

        info: dict
            a dict extracted from the tdf file reading with the info
            required to extract Point3D data from it.

        Returns
        -------
        points: dict
            a dict with all the tracks provided as SimbioPy.Vector objects.
        """
        # check if the file has to be read by frame or by track
        if not info["Format"] in [5]:
            raise IOError("Invalid 'Format' info {}".format(info["Format"]))

        # get the file read (tracks and frames are inverted)
        fid, n_tracks, n_frames, freq, time = readFile(file, info["Offset"])

        # read the data
        fid.seek(2, 1)
        tracks, labels, index = readTracks(
            fid, n_frames, n_tracks, freq, time, False, 9
        )

        # generate the output dict
        accs = {}
        gyrs = {}
        for i, label in enumerate(labels):
            acc_cols = np.arange(3) + 3 * i
            accs[label] = Vector(
                data=tracks[:, acc_cols],  # m/s^2
                columns=["X", "Y", "Z"],
                index=index,
            )
            gyr_cols = acc_cols + 3
            gyrs[label] = Vector(
                data=tracks[:, gyr_cols] * 180 / np.pi,  # deg/s
                columns=["X", "Y", "Z"],
                index=index,
            )

        return {"Acceleration3D": accs, "AngularVelocity3D": gyrs}

    def resize(
        ref: pd.DataFrame,
        reset_time: bool = True,
        **kwargs: Tuple[Vector, pd.DataFrame]
    ) -> dict:
        """
        resize the data contained in kwargs to match the sample range
        of ref.

        Paramters
        ---------
        ref: Vector
            a vector containing the reference data time.

        reset_time: bool
            if True the time of all the returned array will start from zero.

        kwargs: key-values vectors
            a variable number of named vectors to be resized according to ref.

        Returns
        -------
        resized: dict
            a dict containing all the dataframes passed as kwargs resized
            according to ref.
        """

        # check the entries
        txt = "{} must be a pandas DataFrame object."
        assert isinstance(ref, pd.DataFrame), txt.format("'ref'")
        for key, df in kwargs.items():
            assert isinstance(df, pd.DataFrame), txt.format(key)
        assert reset_time or not reset_time, "'reset_time' must be a bool object."

        # get the start and end ref time
        start = np.min(ref.index.to_numpy())
        stop = np.max(ref.index.to_numpy())

        # resize all data
        idx = {i: v.index.to_numpy() for i, v in kwargs.items()}
        valid = {i: np.where((v >= start) & (v <= stop))[0] for i, v in idx.items()}
        resized = {i: v.iloc[valid[i]] for i, v in kwargs.items()}

        # check if the time has to be reset
        if reset_time:
            for i in resized:
                resized[i].index = pd.Index(resized[i].index.to_numpy() - start)

        return resized

    # ----
    # MAIN
    # ----

    # codes
    tdf_signature = "41604B82CA8411D3ACB60060080C6816"
    ids = {
        5: {"fun": getPoint3D, "label": "Point3D"},
        12: {"fun": getForce3D, "label": "Force3D"},
        11: {"fun": getEMG, "label": "EMG"},
        17: {"fun": getIMU, "label": "IMU"},
    }

    # read the file
    fid = open(path, "rb")

    # check the signature
    sig = ["{:08x}".format(b) for b in struct.unpack("IIII", fid.read(16))]
    sig = "".join(sig)
    if sig != tdf_signature.lower():
        raise IOError("invalid file")

    # get the number of entries
    _, n_entries = struct.unpack("Ii", fid.read(8))
    if n_entries <= 0:
        raise IOError("The file specified contains no data.")

    # check each entry to find the available blocks
    next_entry_offset = 40
    blocks = []
    for _ in range(n_entries):

        if -1 == fid.seek(next_entry_offset, 1):
            raise IOError("Error: the file specified is corrupted.")

        # get the data types
        block_info = struct.unpack("IIii", fid.read(16))
        bi = {i: v for i, v in zip(["Type", "Format", "Offset", "Size"], block_info)}

        # retain only valid block types
        if any([i == bi["Type"] for i in ids]):
            blocks += [{"fun": ids[bi["Type"]]["fun"], "info": bi}]

        # update the offset
        next_entry_offset = 272

    # close the file
    fid.close()

    # read the available data
    out = {}
    for b in blocks:
        for key, value in b["fun"](path, b["info"]).items():
            if not any([i == key for i in out]):
                out[key] = {}
            out[key].update(value)

    # resize the data to kinematics (where appropriate)
    has_kinematics = any([i in ["Point3D"] for i in out])
    if fit_to_kinematics and has_kinematics:
        valid = {i: v.dropna() for i, v in out["Point3D"].items()}
        start = np.min([np.min(v.index.to_numpy()) for _, v in valid.items()])
        stop = np.max([np.max(v.index.to_numpy()) for _, v in valid.items()])
        ref = out["Point3D"][[i for i in out["Point3D"]][0]]
        idx = ref.index.to_numpy()
        idx = np.where((idx >= start) & (idx <= stop))[0]
        ref = ref.iloc[idx]
        out = {l: resize(ref, True, **v) for l, v in out.items()}

    # return what has been read
    return out


def three_points_angle(a: Vector, b: Vector, c: Vector, name: str = None):
    """
    return the angle between 3 points using the cosine theorem.

    Parameters
    ----------
    a, b, c: Vector
        the vector objects.

    name: str or None
        the name of the column in the output dataframe

    Returns
    -------
    q: Vector
        the angle in radiants.
    """

    # check the data
    if name is None:
        name = "Angle"
    assert isinstance(name, str), "name must be a string"
    assert a.matches(b), "a does not match b"
    assert a.matches(c), "a does not match c"
    assert b.matches(c), "b does not match c"

    # get the segments
    ab = (b - a).norm.values.flatten()
    bc = (b - c).norm.values.flatten()
    ac = (c - a).norm.values.flatten()

    # return the angle
    q = np.arccos((ac ** 2 - ab ** 2 - bc ** 2) / (-2 * ab * bc))
    return Vector(q, columns=[name], index=a.index)


def point_on_line_projection(A: Vector, B: Vector, C: Vector):
    """
    return the coordinates of O, i.e. the projection of A along the
    line passing through B and C.

    Parameters
    ----------
    A: Vector
        the point not included on the BC line.

    B: Vector
        a vector with dimensions and index matchable with A.

    C: Vector
        another vector with dimensions and index matchable with A.

    Returns
    -------
    O:  Vector
        the vector being along the BC line and having minimum distance
        from A.
    """

    # check the entries
    assert C.matches(A), "A, B and C must be matchable vectors."
    assert C.matches(B), "A, B and C must be matchable vectors."

    # get the angle ACB and from that calculate the length
    # distance of O from C
    alpha = three_points_angle(A, C, B)
    u = (A - C).norm * np.cos(alpha.values)

    # use the distance u to find O along the CB segment
    n = np.ones(A.shape)
    return (B - C) / (n * (B - C).norm.values) * (n * u.values) + C
