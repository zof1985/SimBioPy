# IMPORTS

import os
import struct
import numpy as np
import pandas as pd


# METHODS


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
        vd[v] = pd.DataFrame(
            data=np.vstack(np.atleast_2d(coordinates)).T,
            index=time,
            columns=nn,
        )

    return vd


def read_tdf(path):
    """
    Return the readings from a .tdf file as dicts of Vectors objects.

    Parameters
    ----------
    path: str
        an existing emt path.

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
            points[labels[trk]] = pd.DataFrame(
                data=tracks[:, cols],
                columns=["X", "Y", "Z"],
                index=index,
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
            points[labels[trk]] = pd.DataFrame(
                data=tracks[:, point_cols],
                columns=["X", "Y", "Z"],
                index=index,
            )
            force_cols = np.arange(3 * (trk + 1), 3 * (trk + 1) + 3)
            forces[labels[trk]] = pd.DataFrame(
                data=tracks[:, force_cols],
                columns=["X", "Y", "Z"],
                index=index,
            )
            moment_cols = np.arange(3 * (trk + 2), 3 * (trk + 2) + 3)
            moments[labels[trk]] = pd.DataFrame(
                data=tracks[:, moment_cols],
                coordinates=["X", "Y", "Z"],
                index=index,
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
            channels[lbl] = pd.DataFrame(
                data=np.atleast_2d(trk).T,
                columns=[lbl],
                index=index,
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
        return (vector - ov).dot(self.orientation.T)

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
        return vector.dot(self.orientation) + ov

    def _gram_schmidt(self, vectors):
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


@pd.api.extensions.register_dataframe_accessor("vector")
class Vector:
    def __init__(self, pandas_obj):
        self._obj = pandas_obj

    @property
    def sampling_frequency(self):
        """
        return the "average" sampling frequency of the Vector.
        """
        return float(1 / np.mean(np.diff(self._obj.index.to_numpy())))

    def norm(self):
        """
        get the norm of the vector.
        """
        return pd.DataFrame(
            data=np.sqrt(np.sum(self._obj.values ** 2, axis=1)),
            columns="||{}||".format("+".join(self._obj.columns.to_numpy())),
            index=self._obj.index,
        )
