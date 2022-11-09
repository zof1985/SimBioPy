# BTS BIOENGINEERING IMPORTING MODULE


#! IMPORTS


from .sensors import *
from .geometry import *
from .models import *
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
    values = values.astype(np.float32)

    # get the columns of interest
    cols = np.arange(np.argwhere(vrs == "Time").flatten()[0] + 1, len(vrs))

    # get the rows in the data to be extracted
    rows = np.argwhere(np.any(~np.isnan(values[:, cols]), 1)).flatten()
    rows = np.arange(np.min(rows), np.max(rows) + 1)

    # get time
    time = values[rows, 1].flatten()

    # get the unit of measurement
    unit = lines[3][1]

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
            unit=unit,
        )

    return vd


def read_tdf(path: str, fit_to_kinematics: bool = False):
    """
    Return the readings from a .tdf file as dicts of 3D objects.

    Parameters
    ----------
    path: str
        an existing tdf path.

    fit_to_kinematics: bool
        should the data be resized according to kinematics readings?
        if True, all data are fit such as the start and end of the
        data match with the start and end of the kinematic data.

    Returns
    -------
    a dict containing the distinct data properly arranged by type.
    """
    return TDF(path, fit_to_kinematics).Model3D


class TDF:
    """
    object defining a TDF file object.

    Parameters
    ----------
    path: str
        the root to a valid tdf file.

    fit_to_kinematics: bool
        should the data be resized according to kinematics readings?
        if True, all data are fit such as the start and end of the
        data match with the start and end of the kinematic data.
    """

    # ****** VARIABLES ******#

    _path = None
    _fit_to_kinematics = None
    _blocks = None
    _fid = None
    _tdf_signature = "41604B82CA8411D3ACB60060080C6816"
    _Model3D = None

    # ****** CONSTRUCTOR ******#

    def __init__(self, path: str, fit_to_kinematics: bool = True):

        # check the validity of the entered path
        assert os.path.exists(path), path + " does not exist."
        assert path[-4:] == ".tdf", path + ' must be an ".tdf" path.'
        txt = "'fit_to_kinematics' must be a bool instance."
        assert fit_to_kinematics or not fit_to_kinematics, txt
        self._path = path
        self._fit_to_kinematics = fit_to_kinematics

        # check the available data
        self._fid = open(path, "rb")

        # check the signature
        sig = struct.unpack("IIII", self._fid.read(16))
        sig = "".join(["{:08x}".format(b) for b in sig])
        if sig != self._tdf_signature.lower():
            raise IOError("invalid file")

        # get the number of entries
        _, n_entries = struct.unpack("Ii", self._fid.read(8))
        if n_entries <= 0:
            raise IOError("The file specified contains no data.")

        # reference indices
        ids = {
            5: {
                "fun": self._get_Marker3D,
                "label": "Marker3D",
            },
            7: {
                "fun": self._get_ForcePlatformCalibrationData,
                "label": "ForcePlatform3DCalibration",
            },
            # 9: {
            #     "fun": self._get_UntrackedForce3D,
            #     "label": "ForcePlatform3D",
            # },
            12: {
                "fun": self._get_Force3D,
                "label": "ForcePlatform3D",
            },
            11: {
                "fun": self._getEMG,
                "label": "EMG",
            },
            17: {
                "fun": self._getIMU,
                "label": "IMU",
            },
        }

        # check each entry to find the available blocks
        next_entry_offset = 40
        self._blocks = []
        for _ in range(n_entries):

            if -1 == self._fid.seek(next_entry_offset, 1):
                raise IOError("Error: the file specified is corrupted.")

            # get the data types
            block_info = struct.unpack("IIii", self._fid.read(16))
            block_labels = ["Type", "Format", "Offset", "Size"]
            bi = dict(zip(block_labels, block_info))

            # retain only valid block types
            if any([i == bi["Type"] for i in ids]):
                self._blocks += [{"fun": ids[bi["Type"]]["fun"], "info": bi}]

            # update the offset
            next_entry_offset = 272

        # read the available data
        out = {}
        for b in self._blocks:
            if b["info"]["Type"] != 7:
                for key, value in b["fun"](b["info"]).items():
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
            out = self._resize(out, ref, True)

        # return what has been read
        self._Model3D = Model3D()
        for elems in out.values():
            self._Model3D.append(**elems)

        # close the file
        self._fid.close()

    # ****** GETTERS ******#

    @property
    def Model3D(self):
        """return the Model3D object representing the tdf file."""
        return self._Model3D

    @property
    def fit_to_kinematics(self):
        """
        returns True if the model has been resized according to kinematic data.
        """
        return self._fit_to_kinematics

    @property
    def path(self):
        """returns path to the file"""
        return self._path

    # ****** METHODS ******#

    def _read_tracks(
        self,
        n_frames: int,
        n_tracks: int,
        freq: int,
        time: float,
        by_frame: bool,
        size: int,
        has_labels: bool = True,
    ):
        """
        internal method used to extract 3D tracks from tdf file.

        Parameters
        ----------
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

        has_labels: bool
            used on untracked force platforms to avoid reading labels.

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
            if has_labels:
                lbls = struct.unpack("256B", self._fid.read(256))
                lbls = tuple(chr(i) for i in lbls)
                labels[trk] = "".join(lbls).split(chr(0), 1)[0]

            # read data
            if by_frame:
                n = size * n_tracks * n_frames
                segments = struct.unpack("%if" % n, self._fid.read(n * 4))
                tracks = np.array(segments)
                tracks = tracks.reshape(n_frames, size * n_tracks).T

            # read by track
            else:
                n = struct.unpack("i", self._fid.read(4))[0]
                self._fid.seek(4, 1)
                segments = struct.unpack(f"{2 * n}i", self._fid.read(8 * n))
                segments = np.array(segments).reshape(n, 2).T
                cols = np.atleast_2d(np.arange(size) + size * trk)
                for s in np.arange(n):
                    for r in np.arange(segments[1, s]) + segments[0, s]:
                        vals = self._fid.read(4 * size)
                        if r < tracks.shape[0]:
                            tracks[r, cols] = struct.unpack(f"{size}f", vals)

        # calculate the index (in msec)
        idx = np.round((np.arange(n_frames) / freq + time) * 1000).astype(int)

        # return the tracks
        return tracks, labels, idx

    def _get_Marker3D(self, info: dict):
        """
        read Marker3D tracks data from the provided tdf file.

        Paramters
        ---------
        info: dict
            a dict extracted from the tdf file reading with the info
            required to extract Point3D data from it.

        Returns
        -------
        out: dict
            a dict of Marker3D and Link3D objects.
        """

        # get the file read
        self._fid.seek(info["Offset"], 0)
        frames, freq, time, n_tracks = struct.unpack("iifi", self._fid.read(16))

        # calibration data (read but not exported)
        _ = np.array(struct.unpack("3f", self._fid.read(12)))
        _ = np.array(struct.unpack("9f", self._fid.read(36))).reshape(3, 3).T
        _ = np.array(struct.unpack("3f", self._fid.read(12)))
        self._fid.seek(4, 1)

        # check if links exists
        if info["Format"] in [1, 3]:
            (n_links,) = struct.unpack("i", self._fid.read(4))
            self._fid.seek(4, 1)
            links = struct.unpack(
                "%ii" % (2 * n_links),
                self._fid.read(8 * n_links),
            )
            links = np.reshape(links, (len(links) // 2, 2))

        # check if the file has to be read by frame or by track
        by_frame = info["Format"] in [3, 4]
        by_track = info["Format"] in [1, 2]
        if not by_frame and not by_track:
            raise IOError("Invalid 'Format' info {}".format(info["Format"]))

        # read the data
        tracks, labels, index = self._read_tracks(
            frames,
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

    def _get_Force3D(self, info: dict):
        """
        read Force3D tracks data from the provided tdf file.

        Paramters
        ---------
        info: dict
            a dict extracted from the tdf file reading with the info
            required to extract Point3D data from it.

        Returns
        -------
        fp: dict
            a dict of sensors.ForcePlatform3D objects.
        """
        # get the file read
        self._fid.seek(info["Offset"], 0)
        n_tracks, freq, time, frames = struct.unpack("iifi", self._fid.read(16))

        # calibration data (read but not exported)
        _ = np.array(struct.unpack("3f", self._fid.read(12)))
        _ = np.array(struct.unpack("9f", self._fid.read(36))).reshape(3, 3).T
        _ = np.array(struct.unpack("3f", self._fid.read(12)))
        self._fid.seek(4, 1)

        # check if the file has to be read by frame or by track
        by_frame = info["Format"] in [2]
        by_track = info["Format"] in [1]
        if not by_frame and not by_track:
            raise IOError("Invalid 'Format' info {}".format(info["Format"]))

        # read the data
        tracks, labels, index = self._read_tracks(
            frames, n_tracks, freq, time, by_frame, 9
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

    def _get_ForcePlatformCalibrationData(self, info: dict):
        """
        read Force3D tracks data from the provided tdf file.

        Paramters
        ---------
        info: dict
            a dict extracted from the tdf file reading with the info
            required to extract Point3D data from it.

        Returns
        -------
        points: dict
            a dict with all the tracks provided as simbiopy.ForcePlatform3D
            objects.
        """
        # get the basic info
        self._fid.seek(info["Offset"], 0)
        n_plats = struct.unpack("i", self._fid.read(4))[0]
        self._fid.seek(4, 1)

        # get the platforms map
        platforms = struct.unpack(
            "".join(np.tile("h", n_plats)),
            self._fid.read(n_plats * 2),
        )

        # read the calibration data
        cal_data = {}
        fmt = "{:0" + str(round(np.ceil(np.log10(len(platforms))))) + "d}"
        for platform in platforms:
            label = struct.unpack("256B", self._fid.read(256))
            size = struct.unpack("2f", self._fid.read(8))
            position = struct.unpack("12f", self._fid.read(48))
            self._fid.seek(256, 1)
            cal_data["ForcePlatform" + fmt.format(platform + 1)] = {
                "Label": "".join([chr(i) for i in label]).split(chr(0))[0],
                "Size": tuple(np.round(size, 3)),
                "Position": np.round(position, 3).reshape((4, 3)),
            }

        return cal_data

    def _get_UntrackedForce3D(self, info: dict):
        """
        read Force3D tracks data from the provided tdf file.

        Paramters
        ---------
        info: dict
            a dict extracted from the tdf file reading with the info
            required to extract Point3D data from it.

        Returns
        -------
        fp: dict
            a dict of sensors.ForcePlatform3D objects.
        """
        # obtain the force calibration data
        cal_block = [i["info"] for i in self._blocks if i["info"]["Type"] == 7]
        if len(cal_block) == 0:
            txt = "No calibration data have been found for force platforms."
            raise TypeError(txt)
        calibration_data = self._get_ForcePlatformCalibrationData(cal_block[0])

        # read the file
        self._fid.seek(info["Offset"], 0)
        tracks, freq, time, frames = struct.unpack("iifi", self._fid.read(16))
        plat_map = struct.unpack(
            "".join(np.tile("h", tracks)),
            self._fid.read(tracks * 2),
        )

        # check if the file has to be read by frame or by track
        by_frame = info["Format"] in [2]
        by_track = info["Format"] in [1]
        if not by_frame and not by_track:
            raise IOError("Invalid 'Format' info {}".format(info["Format"]))

        # read the data
        tracks, _, index = self._read_tracks(
            frames,
            tracks,
            freq,
            time,
            by_frame,
            6,
            False,
        )

        # generate the output dict
        fp = {}
        fmt = "{:0" + str(round(np.ceil(np.log10(len(plat_map))))) + "d}"
        for platform in plat_map:
            label = "ForcePlatform" + fmt.format(platform + 1)
            _, size, position = calibration_data[label].values()
            points = np.zeros((tracks.shape[0], 3))
            points[:, 0] = tracks[:, 6 * platform]
            points[:, 2] = tracks[:, 6 * platform + 1]
            forces = tracks[:, 6 * platform + np.arange(3) + 2]
            My = tracks[:, 6 * platform + 5]
            moments = np.zeros_like(forces)
            fp[label] = ForcePlatform3D(
                force=forces,
                moment=moments,
                origin=points,
                index=index,
                force_unit="N",
                moment_unit="Nm",
                origin_unit="m",
            )

        return {"ForcePlatform": fp}

    def _getEMG(self, info: str):
        """
        read EMG tracks data from the provided tdf file.

        Paramters
        ---------
        info: dict
            a dict extracted from the tdf file reading with the info
            required to extract Point3D data from it.

        Returns
        -------
        channels: dict
            a dict with all the EMG channels provided as simbiopy.EmgSensor.
        """
        # get the file read
        self._fid.seek(info["Offset"], 0)
        tracks, freq, time, frames = struct.unpack("iifi", self._fid.read(16))

        # check if the file has to be read by frame or by track
        by_frame = info["Format"] in [2]
        by_track = info["Format"] in [1]
        if not by_frame and not by_track:
            raise IOError("Invalid 'Format' info {}".format(info["Format"]))

        # read the data
        self._fid.read(tracks * 2)
        tracks, labels, index = self._read_tracks(
            frames,
            tracks,
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

    def _getIMU(self, info: str):
        """
        read IMU tracks data from the provided tdf file.

        Paramters
        ---------
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

        # get the file read
        self._fid.seek(info["Offset"], 0)
        tracks, frames, freq, time = struct.unpack("iifi", self._fid.read(16))

        # read the data
        self._fid.seek(2, 1)
        tracks, labels, index = self._read_tracks(frames, tracks, freq, time, False, 9)

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

    def _resize(self, obj, ref, reset_time=True):
        """
        resize the data contained in kwargs to match the sample range
        of ref.

        Paramters
        ---------
        obj: Any
            the object to be resized

        ref: GeometricObject
            a point containing the reference data time.

        reset_time: bool
            if True the time of all the returned array will start from zero.

        Returns
        -------
        resized: dict
            a dict containing all the objects passed as kwargs resized
            according to ref.
        """

        # check the entries
        txt = "{} must be a simbiopy.GeometricObject instance."
        assert isinstance(ref, GeometricObject), txt.format("'ref'")
        txt = "'reset_time' must be a bool object."
        assert reset_time or not reset_time, txt

        # get the start and end ref time
        start = np.min(ref.index)
        stop = np.max(ref.index)

        # resize all data
        def _resize(obj):
            if isinstance(obj, dict):
                return {i: _resize(v) for i, v in obj.items()}
            idx = np.where((obj.index >= start) & (obj.index <= stop))[0]
            out = obj.iloc[idx]
            if reset_time:
                for attr in out._attributes:
                    df = getattr(out, attr)
                    idx = df.index.to_numpy() - start
                    df.index = pd.Index(np.round(idx).astype(int))
                    setattr(out, attr, df)
            return out

        return _resize(obj)
