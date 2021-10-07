# IMPORTS

import os
import struct
import base
import numpy as np
import pandas as pd


# METHODS


def gram_schmidt(vectors):
    """
    Return the orthogonal basis defined by a set of vectors using the
    Gram-Schmidt algorithm.

    Parameters:
        vectors (np.ndarray): a NxN numpy.ndarray to be orthogonalized (by row).

    Returns:
        a NxN numpy.ndarray containing the orthogonalized arrays.
    """

    # internal functions to simplify the calculation
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


def to_vector(df):
    """
    Try to convert a standard pandas DataFrame into a Vector.

    Parameters
    ----------

    df: pd.DataFrame
        the dataframe to be converted.
        It must have columns as "X (Y)" where 'X' is the dimension name and 'Y' is the unit
        of measurement.

    Returns
    -------

    vec: Vector
        a Vector object with the reported data.
    """
    data_unit = df.columns[0].split(" ")[-1][1:-1]
    index_unit = df.index[0].split(" ")[-1][1:-1]
    columns = [i.split(" ")[0] for i in df.columns.to_numpy()]
    index = [float(i.split(" ")[0]) for i in df.index.to_numpy()]
    return Vector(
        data=df.values,
        columns=columns,
        index=index,
        data_unit=data_unit,
        index_unit=index_unit,
    )


def read_csv(path):
    """
    Create a "Vector" from a "csv" path.

    Parameters
    ----------

    path: (str)
        an existing ".csv" or "txt" path or a folder containing csv files.
        The files must contain the index in the first column and the others must be stored
        "X (Y)" where 'X' is the dimension name and 'Y' is the unit of measurement.

    Returns
    -------

    vec: Vector
        a Vector object with the reported data.
    """

    # check the validity of the entered path
    assert os.path.exists(path), path + " does not exist."
    assert path[-4:] == ".csv" or path[-4:] == ".txt", (
        path + ' must be a ".csv" or ".txt" file.'
    )

    # return the data
    return to_vector(pd.read_csv(path, index_col=0))


def read_excel(path, sheets=None, exclude_errors=True):
    """
    Create a dict of Vector objects from an excel path.

    Parameters
    ----------

    path: str
        an existing excel path.
        The sheets must contain the index in the first column and the others must be stored
        "X (Y)" where 'X' is the dimension name and 'Y' is the unit of measurement.

    sheets: str
        the sheets to be imported. In None, all sheets are imported.

    exclude_errors: bool
        If a sheet generates an error during the import would you like to skip it and import
        the others?

    Returns
    -------

    vectors: dict
        a dict with the imported vectors.
    """

    # check the validity of the entered path
    assert os.path.exists(path), path + " does not exist."
    assert path[-5:] == ".xlsx" or path[-4:] == ".xls", (
        path + ' must be a ".xlsx" or ".xls" file.'
    )

    # get the sheets
    dfs = base.from_excel(path, sheets)

    # import the sheets
    vd = {}
    for i in dfs:
        try:
            vd[i] = to_vector(dfs[i])
        except Exception:
            if not exclude_errors:
                break
    return vd


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

    # get the units
    unit = lines[3][1]

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
            data_unit=unit,
            index_unit="s",
        )

    return vd


def read_tdf(path, point_unit="m", force_unit="N", moment_unit="Nm", emg_unit="uV"):
    """
    Return the readings from a .tdf file as dicts of Vectors objects.

    Parameters
    ----------
    path: str
        an existing emt path.

    point_unit, force_unit, moment_unit, emg_unit: str
        the unit of measurement for the various data types

    Returns
    -------
    points: dict
        a dict with the Point 3D tracks vectors.

    forces: dict
        a dict with the Force 3D tracks vectors.

    moments: dict
        a dict with the Torque 3D tracks vectors.

    emgs: dict
         a dict with the EMG activity channels as vectors.
    """

    # check the validity of the entered path
    assert os.path.exists(path), path + " does not exist."
    assert path[-4:] == ".tdf", path + ' must be an ".tdf" path.'

    # file reader with basic info
    def read_file(file, offset):
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

        # update the reader
        fid.seek(offset)

        # get the basic info
        base_info = struct.unpack("iifi", fid.read(16))
        rows, freq, time, cols = base_info

        return fid, rows, cols, freq, time

    # generic track extractor
    def read_tracks(fid, n_frames, n_tracks, freq, time, by_frame, size):
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

            # read by frame
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

    # camera tracks
    def getPoint3D(file, info):
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
        fid, n_frames, n_tracks, freq, time = read_file(file, info["Offset"])

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
        tracks, labels, index = read_tracks(
            fid, n_frames, n_tracks, freq, time, by_frame, 3
        )

        # generate the output dict
        points = {}
        for trk in range(n_tracks):
            cols = np.arange(3 * trk, 3 * trk + 3)
            points[labels[trk]] = Vector(
                data=tracks[:, cols],
                columns=["X", "Y", "Z"],
                index=index,
                data_unit=point_unit,
                index_unit="s",
            )
        return points

    # force platform data
    def getForce3D(file, info):
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
        fid, n_tracks, n_frames, freq, time = read_file(file, info["Offset"])

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
        tracks, labels, index = read_tracks(
            fid, n_frames, n_tracks, freq, time, by_frame, 9
        )

        # generate the output dict
        points = {}
        forces = {}
        moments = {}
        for trk in range(n_tracks):
            point_cols = np.arange(3 * trk, 3 * trk + 3)
            points[labels[trk]] = Vector(
                data=tracks[:, point_cols],
                columns=["X", "Y", "Z"],
                index=index,
                data_unit=point_unit,
                index_unit="s",
            )
            force_cols = np.arange(3 * (trk + 1), 3 * (trk + 1) + 3)
            forces[labels[trk]] = Vector(
                data=tracks[:, force_cols],
                columns=["X", "Y", "Z"],
                index=index,
                data_unit=force_unit,
                index_unit="s",
            )
            moment_cols = np.arange(3 * (trk + 2), 3 * (trk + 2) + 3)
            moments[labels[trk]] = Vector(
                data=tracks[:, moment_cols],
                coordinates=["X", "Y", "Z"],
                index=index,
                data_unit=moment_unit,
                index_unit="s",
            )
        fid.close()
        return points, forces, moments

    # read EMG data
    def getEMG(file, info):
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
            a dict with all the EMG channels provided as SimbioPy.Vector objects.
        """
        # get the file read (tracks and frames are inverted here)
        fid, n_tracks, n_frames, freq, time = read_file(file, info["Offset"])

        # get the EMG channels map (unused its nTracks * int16 element)
        _ = fid.read(n_tracks * 2)

        # check if the file has to be read by frame or by track
        by_frame = info["Format"] in [2]
        by_track = info["Format"] in [1]
        if not by_frame and not by_track:
            raise IOError("Invalid 'Format' info {}".format(info["Format"]))

        # read the data
        tracks, labels, index = read_tracks(
            fid, n_frames, n_tracks, freq, time, by_frame, 1
        )

        # generate the output dict
        channels = {}
        for trk, lbl in zip(tracks.T, labels):
            channels[lbl] = Vector(
                data=np.atleast_2d(trk).T,
                columns=[lbl],
                index=index,
                data_unit=emg_unit,
                index_unit="s",
            )
        return channels

    # codes
    tdf_signature = "41604B82CA8411D3ACB60060080C6816"
    id_point_3d = 5
    id_force_3d = 12
    id_emg = 11

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
    blocks = {}
    for _ in range(n_entries):

        # corrupted file
        if -1 == fid.seek(next_entry_offset, 1):
            raise IOError("Error: the file specified is corrupted.")

        # get the data types
        block_info = struct.unpack("IIii", fid.read(16))
        bi = {
            "Type": block_info[0],
            "Format": block_info[1],
            "Offset": block_info[2],
            "Size": block_info[3],
        }

        # check the type of block
        if bi["Type"] == id_point_3d:
            blocks["Point3D"] = bi
        elif bi["Type"] == id_force_3d:
            blocks["Force3D"] = bi
        elif bi["Type"] == id_emg:
            blocks["EMG"] = bi

        # update the offset
        next_entry_offset = 272

    # close the file
    fid.close()

    # read the available data
    points = {}
    forces = {}
    moments = {}
    emgs = {}
    for key, info in blocks.items():
        if key == "Point3D":
            points.update(**getPoint3D(path, info))
        elif key == "Force3D":
            pnt, frz, mnt = getForce3D(path, info)
            points.update(**pnt)
            forces.update(**frz)
            moments.update(**mnt)
        elif key == "EMG":
            emgs.update(**getEMG(path, info))

    # return what has been read
    return points, forces, moments, emgs


# CLASSES


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

        # handle the origin parameter
        assert isinstance(
            origin, (list, np.ndarray)
        ), "origin must be a list or numpy array."
        self.origin = np.atleast_1d(origin).flatten()

        # handle the names parameter
        if names is None:
            names = ["X{}".format(i + 1) for i in range(len(self.origin))]
        assert isinstance(
            names, (list, np.ndarray)
        ), "names must be None, list or a numpy array."
        self.names = np.atleast_1d(names).flatten()

        # handle the orientation parameter
        if orientation is None:
            orientation = np.eye(len(self.origin))
        assert isinstance(
            orientation, (list, np.ndarray)
        ), "orientation must be a list or numpy array."
        self.orientation = gram_schmidt(orientation)

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

        if not isinstance(vector, Vector):
            return False
        if len(self.origin) != vector.coordinates.shape[1]:
            return False
        if not np.all([np.any(vector.names == i) for i in self.names]):
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
        vec = vector - ov
        vec.coordinates[:, :] = vec.coordinates.dot(self.orientation.T)
        return vec

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
        vec = vector.copy()
        ov = np.vstack([np.atleast_2d(self.origin) for _ in range(vector.shape[0])])
        vec.coordinates[:, :] = vec.coordinates.dot(self.orientation) + ov
        return vec


@pd.api.extensions.register_dataframe_accessor("vector")
class Vector:
    def __init__(self, pandas_obj, data_unit, index_unit):
        self._obj = pandas_obj
        self.data_unit = data_unit
        self.index_unit = index_unit

    def copy(self):
        return Vector(
            data=self.values,
            columns=self.columns,
            index=self.index,
            data_unit=self.data_unit,
            index_unit=self.index_unit,
        )

    def to_df(self):
        return pd.DataFrame(
            data=self.values,
            columns=pd.Index(
                [i + " ({})".format(self.data_unit) for i in self.columns]
            ),
            index=pd.Index(
                [str(i) + " ({})".format(self.index_unit) for i in self.index]
            ),
        )

    def __str__(self):
        """
        printing function
        """
        self.to_df().__str__()

    def __repr__(self):
        """
        representation function
        """
        return self.__str__()

    @property
    def sampling_frequency(self):
        """
        return the "average" sampling frequency of the Vector.
        """
        return float(1 / np.mean(np.diff(self.index.to_numpy())))

    def rotate(self, ref):
        """
        rotate the current marker to the provided reference frame

        Parameters
        ----------
        ref: ReferenceFrame
            the target reference frame.

        Returns
        -------
        rot: Vector
            self rotated into ref.
        """
        assert isinstance(ref, ReferenceFrame), "ref must be a ReferenceFrame object."
        return ref.rotate(self)

    def invert(self, ref):
        """
        invert the rotation generated by the provided reference frame

        Parameters
        ----------
        ref: ReferenceFrame
            the target reference frame.

        Returns
        -------
        rot: Vector
            self rotated from ref.
        """
        assert isinstance(ref, ReferenceFrame), "ref must be a ReferenceFrame object."
        return ref.invert(self)

    def cross(self, value):
        """
        get the cross product between 3D Vectors.

        Parameters
        ----------
        value: Vector
            a 3D Vector to be used for the cross product operation

        Returns
        -------
        crossed: Vector
            the result of the cross product operation.
        """
        txt = "cross operator is valid only between 3D Vectors."
        assert isinstance(value, Vector), txt
        assert len(self.names) == len(value.names) == 3, txt
        vec = self.copy()
        vec.coordinates = np.cross(vec.coordinates, value.coordinates)
        return vec

    def norm(self):
        """
        get the norm of the vector.
        """
        return Vector(
            data=np.sqrt(np.sum(self.values ** 2, axis=1)),
            columns="||{}||".format("+".join(self.names)),
            index=self.index,
            data_unit=self.data_unit,
            index_unit=self.index_unit,
        )

    def to_csv(self, path):
        """
        Store the current Vector data as csv path.

        Parameters
        ----------
        path: str
            the path where to store the data.
        """
        self.to_df().to_csv(path)

    def to_excel(self, path, sheet, new_file=False):
        """
        Store the current Vector data as csv path.

        Parameters
        ----------
        path: str
            the path where to store the data.

        sheet: str
            the name of the excel sheet where to store the data

        new_file: bool
            should the saving overwrite exising excel path?
        """
        base.to_excel(
            path=path, df=self.to_df(), sheet=sheet, keep_index=True, new_file=new_file
        )
