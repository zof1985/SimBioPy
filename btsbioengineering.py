# BTS BIOENGINEERING IMPORTING MODULE


#! IMPORTS


from io import BufferedReader
from typing import Tuple
from .sensors import *
from .geometry import *
import os
import struct
import numpy as np
import pandas as pd


#! METHODS


def read_emt(path):
    """
    Create a dict of 3D objects from an emt file.

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
        vd[v] = Point(
            coordinates=np.vstack(np.atleast_2d(coordinates)).T,
            index=time,
            columns=nn,
        )

    return vd


def read_tdf(path: str, fit_to_kinematics: bool = False) -> dict:
    """
    Return the readings from a .tdf file as dicts of 3D objects.

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
    def readFile(
        file: str,
        offset: int,
    ) -> Tuple[BufferedReader, int, int, int, float]:
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
        labels = [""] * n_tracks
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
                tracks = np.array(segments)
                tracks = tracks.reshape(n_frames, size * n_tracks).T

            # read by track
            else:
                (n_seg,) = struct.unpack("i", fid.read(4))
                fid.seek(4, 1)
                segments = struct.unpack(
                    "%ii" % (2 * n_seg),
                    fid.read(8 * n_seg),
                )
                segments = np.array(segments).reshape(n_seg, 2).T
                shape = "{}f".format(size)
                for s in range(n_seg):
                    rng = range(
                        segments[0, s],
                        segments[0, s] + segments[1, s],
                    )
                    for row in rng:
                        val = struct.unpack(shape, fid.read(4 * size))
                        if row < n_frames:
                            cols = np.arange(size * trk, size * trk + size)
                            tracks[row, cols] = val

        # calculate the index
        idx = np.arange(n_frames) / freq + time

        # return the tracks
        fid.close()
        return tracks, labels, idx

    def get_Marker3D(file: str, info: dict) -> dict:
        """
        read Marker3D tracks data from the provided tdf file.

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
            a dict with all the tracks provided as simbiopy.Marker3D objects.
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
            links = struct.unpack(
                "%ii" % (2 * n_links),
                fid.read(8 * n_links),
            )
            links = np.reshape(links, (len(links) // 2, 2))

        # check if the file has to be read by frame or by track
        by_frame = info["Format"] in [3, 4]
        by_track = info["Format"] in [1, 2]
        if not by_frame and not by_track:
            raise IOError("Invalid 'Format' info {}".format(info["Format"]))

        # read the data
        tracks, labels, index = readTracks(
            fid,
            n_frames,
            n_tracks,
            freq,
            time,
            by_frame,
            3,
        )

        # generate the output markers
        points = {}
        for trk in range(n_tracks):
            cols = np.arange(3) + 3 * trk
            points[labels[trk]] = Marker3D(
                coordinates=tracks[:, cols],
                index=index,
                unit="m",
            )

        # generate the links
        lnk = {}
        for link in links:
            p0 = points[labels[link[0]]].coordinates
            p1 = points[labels[link[1]]].coordinates
            lbl = "{} -> {}".format(*[labels[i] for i in link])
            lnk[lbl] = Link3D(p0, p1)

        # get the output
        out = {"Marker3D": points}
        if len(lnk) > 0:
            out["Link3D"] = lnk
        return out

    def get_Force3D(file: str, info: dict) -> Tuple[dict, dict, dict]:
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
            a dict with all the tracks provided as simbiopy.ForcePlatform3D
            objects.
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
        fp = {}
        for trk in range(n_tracks):
            point_cols = np.arange(3) + 9 * trk
            points = tracks[:, point_cols]
            force_cols = np.arange(3) + 3 + 9 * trk
            forces = tracks[:, force_cols]
            moment_cols = np.arange(3) + 6 + 9 * trk
            moments = tracks[:, moment_cols]
            fp[labels[trk]] = ForcePlatform3D(
                force=forces,
                moment=moments,
                origin=points,
                index=index,
                force_unit="N",
                moment_unit="Nm",
                origin_unit="m",
            )

        return {"ForcePlatform": fp}

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
            a dict with all the EMG channels provided as simbiopy.EmgSensor.
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
            fid,
            n_frames,
            n_tracks,
            freq,
            time,
            by_frame,
            1,
        )

        # generate the output
        out = {}
        for i, v in zip(labels, tracks.T):
            out[i] = EmgSensor(amplitude=v * 1e6, index=index, unit="uV")
        return {"EMG": out}

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
            a dict with all the tracks provided as simbiopy.Imu3D objects.
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
        imus = {}
        for i, label in enumerate(labels):
            acc_cols = np.arange(3) + 3 * i
            gyr_cols = acc_cols + 3
            mag_cols = acc_cols + 6
            imus[label] = Imu3D(
                accelerometer=tracks[:, acc_cols],
                gyroscope=tracks[:, gyr_cols],
                magnetometer=tracks[:, mag_cols],
                index=index,
                accelerometer_unit="m/s^2",
                gyroscope_unit="rad/s",
                magnetometer_unit="nT",
            )

        return {"IMU": imus}

    def resize(
        ref: pd.DataFrame,
        reset_time: bool = True,
        **kwargs: Tuple[Point, pd.DataFrame],
    ) -> dict:
        """
        resize the data contained in kwargs to match the sample range
        of ref.

        Paramters
        ---------
        ref: GeometricObject
            a point containing the reference data time.

        reset_time: bool
            if True the time of all the returned array will start from zero.

        kwargs: key-values GeometricObjects
            a variable number of named objects to be resized according to ref.

        Returns
        -------
        resized: dict
            a dict containing all the objects passed as kwargs resized
            according to ref.
        """

        # check the entries
        txt = "{} must be a simbiopy.GeometricObject instance."
        assert isinstance(ref, GeometricObject), txt.format("'ref'")
        for key, obj in kwargs.items():
            assert isinstance(obj, GeometricObject), txt.format(key)
        txt = "'reset_time' must be a bool object."
        assert reset_time or not reset_time, txt

        # get the start and end ref time
        start = np.min(ref.index)
        stop = np.max(ref.index)

        # resize all data
        idx = {i: v.index for i, v in kwargs.items()}
        valid = {}
        for i, v in idx.items():
            valid[i] = np.where((v >= start) & (v <= stop))[0]
        resized = {i: v.iloc[valid[i]] for i, v in kwargs.items()}

        # check if the time has to be reset
        if reset_time:
            for i, v in resized.items():
                for attr in v._attributes:
                    df = getattr(v, attr)
                    df.index = pd.Index(df.index.to_numpy() - start)
                    setattr(resized[i], attr, df)

        return resized

    # ----
    # MAIN
    # ----

    # codes
    tdf_signature = "41604B82CA8411D3ACB60060080C6816"
    ids = {
        5: {"fun": get_Marker3D, "label": "Marker3D"},
        12: {"fun": get_Force3D, "label": "ForcePlatform3D"},
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
        block_labels = ["Type", "Format", "Offset", "Size"]
        bi = dict(zip(block_labels, block_info))

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
    has_kinematics = any([i in ["Marker3D"] for i in out])
    if fit_to_kinematics and has_kinematics:
        valid = {i: v.dropna() for i, v in out["Marker3D"].items()}
        start = np.min([np.min(v.index) for _, v in valid.items()])
        stop = np.max([np.max(v.index) for _, v in valid.items()])
        ref = out["Marker3D"][[i for i in out["Marker3D"]][0]]
        idx = ref.index
        idx = np.where((idx >= start) & (idx <= stop))[0]
        ref = ref.iloc[idx]
        out = {l: resize(ref, True, **v) for l, v in out.items()}

    # return what has been read
    mod = Model3D()
    for elems in out.values():
        for name, obj in elems.items():
            mod.append(obj=obj, name=name)
    return mod
