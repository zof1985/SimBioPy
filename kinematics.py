# IMPORTS

from .base import *
import struct


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


def df2vector(df):
    """
    Try to convert a standard pandas DataFrame into a Vector.

    Parameters
    ----------

    df: pd.DataFrame
        the dataframe to be converted.
        It must have columns as "X (Y)" where 'X' is the dimension name and 'Y' is the unit of measurement.

    Returns
    -------

    vec: Vector
        a Vector object with the reported data.
    """

    unit = np.unique([i.split("(")[-1].split(")")[0] for i in df.columns.to_numpy()])[0]
    index = df.index.to_numpy()
    names = [i.split(" (")[0] for i in df.columns.to_numpy()]
    coordinates = df.values
    return Vector(coordinates=coordinates, names=names, index=index, unit=unit)


def read_csv(path):
    """
    Create a "Vector" from a "csv" path.

    Parameters
    ----------

    path: (str)
        an existing ".csv" or "txt" path or a folder containing csv files.
        The files must contain the index in the first column and the others must be stored "X (Y)" where 'X' is
        the dimension name and 'Y' is the unit of measurement.

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
    return df2vector(pd.read_csv(path, index_col=0))


def read_excel(path, sheets=None, exclude_errors=True):
    """
    Create a dict of Vector objects from an excel path.

    Parameters
    ----------

    path: str
        an existing excel path.
        The sheets must contain the index in the first column and the others must be stored "X (Y)" where 'X' is
        the dimension name and 'Y' is the unit of measurement.

    sheets: str
        the sheets to be imported. In None, all sheets are imported.

    exclude_errors: bool
        If a sheet generates an error during the import would you like to skip it and import the others?

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
    dfs = from_excel(path, sheets)

    # import the sheets
    vd = {}
    for i in dfs:
        try:
            vd[i] = df2vector(dfs[i])
        except Exception:
            if exclude_errors:
                pass
            else:
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
    vars = np.array([i for i in lines[10] if i != ""]).flatten()

    # get the data names
    names = np.unique([i.split(".")[0] for i in vars[2:] if len(i) > 0])

    # get the data values
    values = np.vstack([np.atleast_2d(i[: len(vars)]) for i in lines[11:-2]]).astype(
        float
    )

    # get the columns of interest
    cols = np.arange(np.argwhere(vars == "Time").flatten()[0] + 1, len(vars))

    # get the rows in the data to be extracted
    rows = np.argwhere(np.any(~np.isnan(values[:, cols]), 1)).flatten()
    rows = np.arange(np.min(rows), np.max(rows) + 1)

    # get time
    time = values[rows, 1].flatten()

    # generate a dataframe for each variable
    for v in names:

        # get the dimensions
        D = [i.split(".")[-1] for i in vars if i.split(".")[0] == v]
        D = [""] if len(D) == 1 else D

        # get the data for each dimension
        nn = []
        coordinates = []
        for i in D:
            nn += [i if i != "" else v]
            cols = np.argwhere(vars == v + (("." + i) if i != "" else ""))
            coordinates += [values[rows, cols.flatten()]]

        # setup the output variable
        vd[v] = Vector(
            coordinates=np.vstack(np.atleast_2d(coordinates)).T,
            index=time,
            names=nn,
            unit=unit,
        )

    # return vd
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
    tracks: dict
        a dict with the Point 3D tracks vectors.

    forces: dict
        a dict with the Force 3D tracks vectors.

    muscles: dict
         a dict with the EMG activity channels as vectors.
    """

    # check the validity of the entered path
    assert os.path.exists(path), path + " does not exist."
    assert path[-4:] == ".tdf", path + ' must be an ".tdf" path.'

    class TDF(object):
        """TDF protocol constants"""

        tdfSignature = "41604B82CA8411D3ACB60060080C6816"
        tdfData3DBlockId = 5
        tdfDataGPBlockId = 14

        """ instance members
        marker data
            freqM
            labelsM
            markers
            timeM

        analogue input data
            freqA
            labelsA
            analogue
            timeA
        """

        def __init__(self, fileName):
            try:
                block3d = TDF.validateFile(fileName)
                (
                    self.markers,
                    self.labelsM,
                    self.freqM,
                    nFrames,
                    startTime,
                ) = TDF.get3Ddata(fileName, block3d)
                self.timeM = startTime + np.arange(0, nFrames) / self.freqM
            except Warning as e:
                # data missing
                print(e.args)
            except IOError as e:
                print(e.args)

        @staticmethod
        def validateFile(fileName):
            with open(fileName, "rb") as fid:
                fileSign = "".join(
                    [
                        "{:08x}".format(b4)
                        for b4 in struct.unpack("IIII", fid.read(4 * 4))
                    ]
                )
                if fileSign != TDF.tdfSignature.lower():
                    # print("invalid file")
                    raise IOError("invalid file")

                version, nEntries = struct.unpack("Ii", fid.read(4 * 2))

                if nEntries <= 0:
                    # print('The file specified contains no data.')
                    raise IOError("The file specified contains no data.")

                nextEntryOffset = 40
                # tdfBlockEntries=[]
                block3D, blockIn = None, None
                for i in range(nEntries):
                    if -1 == fid.seek(nextEntryOffset, 1):
                        # print('Error: the file specified is corrupted.')
                        # return None, None
                        raise IOError("Error: the file specified is corrupted.")

                    blockInfo = struct.unpack("IIii", fid.read(4 * 4))
                    BI = {
                        "Type": blockInfo[0],
                        "Format": blockInfo[1],
                        "Offset": blockInfo[2],
                        "Size": blockInfo[3],
                    }

                    if BI["Type"] == 14:
                        blockIn = BI
                    elif BI["Type"] == 5:
                        block3D = BI

                    # tdfBlockEntries.append(BI)
                    nextEntryOffset = 16 + 256

                if block3D is None or block3D["Format"] not in [1, 2]:
                    # print("3D marker data missing")
                    raise Warning("3D marker data missing")

                return block3D

        @staticmethod
        def get3Ddata(fileName, blockInfo):
            with open(fileName, "rb") as fid:
                fid.seek(blockInfo["Offset"])

                nFrames, freq, startTime, nTracks = struct.unpack(
                    "iifi", fid.read(4 * 4)
                )

                # calibration data
                D = np.array(struct.unpack("3f", fid.read(3 * 4)))
                R = np.array(struct.unpack("9f", fid.read(9 * 4))).reshape(3, 3).T
                T = np.array(struct.unpack("3f", fid.read(3 * 4)))

                fid.seek(4, 1)

                if blockInfo["Format"] in [1, 3]:
                    print("this yes")
                    (nLinks,) = struct.unpack("i", fid.read(4))
                    fid.seek(4, 1)
                    links = np.array(
                        struct.unpack("%ii" % (2 * nLinks), fid.read(2 * nLinks * 4))
                    )

                labels = [u""] * nTracks
                tracks = np.ones((nFrames, 3 * nTracks)) * np.nan

                if blockInfo["Format"] in [1, 2]:
                    for trk in range(nTracks):
                        lbl = fid.read(256).decode("mbcs")
                        labels[trk] = str(lbl)
                        # print(labels[trk])
                        # print(fid.read(256))
                        (nSegments,) = struct.unpack("i", fid.read(4))
                        # print("nSegments "+str(nSegments))
                        fid.seek(4, 1)
                        segments = (
                            np.array(
                                struct.unpack(
                                    "%ii" % (2 * nSegments), fid.read(2 * nSegments * 4)
                                )
                            )
                            .reshape(nSegments, 2)
                            .T
                        )
                        # print(segments)
                        for s in range(nSegments):
                            for f in range(
                                segments[0, s], segments[0, s] + segments[1, s]
                            ):
                                tracks[f, 3 * (trk) : 3 * (trk) + 3] = struct.unpack(
                                    "3f", fid.read(3 * 4)
                                )

                elif blockInfo["Format"] in [3, 4]:
                    for trk in range(nTracks):
                        lbl = fid.read(256).decode("mbcs")
                        labels[trk] = str(lbl)
                        # tracks = (fread (fid,[3*nTracks,nFrames],'float32'))';
                        tracks = (
                            np.array(
                                struct.unpack(
                                    "%if" % 3 * nTracks * nFrames,
                                    fid.read(3 * nTracks * nFrames * 4),
                                )
                            )
                            .reshape(nFrames, 3 * nTracks)
                            .T
                        )

                return tracks, labels, freq, nFrames, startTime

    rd = TDF(path)

    # data extractors
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

        # read the file
        fid = open(file, "rb")

        # update the reader
        fid.seek(info["Offset"])

        # get the basic info
        base_info = struct.unpack("iifi", fid.read(16))
        nFrames, freq, startTime, nTracks = base_info

        # calibration data (read but not exported)
        d = np.array(struct.unpack("3f", fid.read(12)))
        r = np.array(struct.unpack("9f", fid.read(36))).reshape(3, 3).T
        t = np.array(struct.unpack("3f", fid.read(12)))
        fid.seek(4, 1)

        # links (read but not used)
        if info["Format"] in [1, 3]:
            (nLinks,) = struct.unpack("i", fid.read(4))
            fid.seek(4, 1)
            links = struct.unpack("%ii" % (2 * nLinks), fid.read(8 * nLinks))
            links = np.array(links)

        # prepare the arrays for the tracks and the labels
        labels = [u""] * nTracks
        tracks = np.ones((nFrames, 3 * nTracks)) * np.nan

        # read the data
        for trk in range(nTracks):

            # get the label
            lbl = str(fid.read(256).decode("mbcs"))
            labels[trk] = "".join([i for i in lbl if i != "\x00"])

            # populate the tracks array according to the block format
            if info["Format"] in [1, 2]:

                (nSeg,) = struct.unpack("i", fid.read(4))
                fid.seek(4, 1)
                segments = struct.unpack("%ii" % (2 * nSeg), fid.read(8 * nSeg))
                segments = np.array(segments).reshape(nSeg, 2).T
                for s in range(nSeg):
                    for row in range(segments[0, s], segments[0, s] + segments[1, s]):
                        cols = np.arange(3) + 3 * trk
                        tracks[row, cols] = struct.unpack("3f", fid.read(12))

            elif info["Format"] in [3, 4]:

                n = 3 * nTracks * nFrames
                segments = struct.unpack("%if" % n, fid.read(n * 4))
                tracks = np.array(segments).reshape(nFrames, 3 * nTracks).T

        # generate the output dict
        points = {}
        for trk in range(nTracks):
            cols = np.arange(3 * trk, 3 * trk + 3)
            points[labels[trk]] = Vector(
                coordinates=tracks[:, cols],
                names=["X", "Y", "Z"],
                index=np.arange(nFrames) / freq + startTime,
                unit="m",
            )
        return points

    # codes
    tdf_signature = "41604B82CA8411D3ACB60060080C6816"
    id_Point3D = 5
    id_Force3D = -1
    id_EMG = -1

    # read the file
    fid = open(path, "rb")

    # check the signature
    sig = ["{:08x}".format(b) for b in struct.unpack("IIII", fid.read(16))]
    sig = "".join(sig)
    if sig != tdf_signature.lower():
        raise IOError("invalid file")

    # get the number of entries
    _, nEntries = struct.unpack("Ii", fid.read(8))
    if nEntries <= 0:
        raise IOError("The file specified contains no data.")

    # check each entry to find the available blocks
    nextEntryOffset = 40
    blocks = {}
    for _ in range(nEntries):

        # corrupted file
        if -1 == fid.seek(nextEntryOffset, 1):
            raise IOError("Error: the file specified is corrupted.")

        # get the data types
        blockInfo = struct.unpack("IIii", fid.read(16))
        bi = {
            "Type": blockInfo[0],
            "Format": blockInfo[1],
            "Offset": blockInfo[2],
            "Size": blockInfo[3],
        }

        # check the type of block
        if bi["Type"] == id_Point3D:
            blocks["Point3D"] = bi
        elif bi["Type"] == id_Force3D:
            blocks["Force3D"] = bi
        elif bi["Type"] == id_EMG:
            blocks["EMG"] = bi

        # update the offset
        nextEntryOffset = 272

    # close the file
    fid.close()

    # read the available data
    for key, info in blocks.items():
        if key == "Point3D":
            blocks[key] = getPoint3D(path, info)
        elif key == "Force3D":
            blocks[key] = getForce3D(path, info)
        elif key == "EMG":
            blocks[key] = getEMG(path, info)

    # return what has been read
    return blocks


# CLASSES


class ReferenceFrame:
    """
    Create a ReferenceFrame instance.

    Attributes
    ----------

    origin: array-like 1D
        a 1D list that contains the coordinates of the ReferenceFrame's origin.

    orientation: NxN array
        a 2D square matrix containing on each row the versor corresponding to each dimension of the origin.

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
            a 2D square matrix containing on each row the versor corresponding to each dimension of the origin.

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
        ov = np.vstack([np.atleast_2d(self.origin) for i in range(vector.shape[0])])
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
        ov = np.vstack([np.atleast_2d(self.origin) for i in range(vector.shape[0])])
        vec.coordinates[:, :] = vec.coordinates.dot(self.orientation) + ov
        return vec


class Vector:
    """
    Create a Vector object instance.
    """

    def __init__(
        self, coordinates, names=None, index=None, sampling_frequency=1.0, unit=""
    ):
        """
        Create a Vector reference object.

        Parameters
        ----------
        coordinates : array-like 2D
            a MxN matrix where each row is a sample and each column a dimension.

        names : array-like 1D
            the list of names defining each dimension. If coordinates is a DataFrame object, then
            names is ignored and the column names of the coordinates dataframe are used.

        index : array-like 1D
            the list of indices for each row of the coordinates. If coordinates is a DataFrame object, then
            index is ignored and the index of the coordinates dataframe are used.

        sampling_frequency: float
            if index is None, and coordinates is not a DataFrame, the sampling frequency is used to generate
            the coordinates' index.

        unit: str
            the unit of measurement of the Vector.
        """

        # check the entries
        txt = "coordinates must be a 2D numpy array or a pandas DataFrame."
        if isinstance(coordinates, pd.DataFrame):
            names = coordinates.columns.to_numpy()
            index = coordinates.index.to_numpy()
            coordinates = coordinates.values
        elif isinstance(coordinates, (list, np.ndarray)):
            coordinates = np.atleast_2d(coordinates)
            if names is None:
                names = ["X{}".format(i + 1) for i in range(coordinates.shape[1])]
            assert isinstance(
                names, (list, np.ndarray)
            ), "names must be a 1D array-like object."
            names = np.atleast_1d(names).flatten()
            if index is None:
                assert isinstance(
                    sampling_frequency, (int, float)
                ), "sampling_frequency must be numeric."
                index = np.linspace(
                    0, sampling_frequency * coordinates.shape[0], coordinates.shape[0]
                )
            else:
                txt = "index must be a 1D array-like object of len = {}".format(
                    coordinates.shape[0]
                )
                assert isinstance(index, (list, np.ndarray)), txt
                index = np.array(index).flatten()
                assert len(index) == coordinates.shape[0], txt
        else:
            ValueError("coordinates must be a 2D numpy array or a pandas DataFrame.")

        # check the unit
        assert isinstance(unit, str), "unit must be a str object."

        # add the data
        setattr(self, "coordinates", np.atleast_2d(coordinates))
        setattr(self, "names", np.array([names]).flatten())
        setattr(self, "index", np.array([index]).flatten())
        setattr(self, "unit", unit)

    def _get_selection(self, item):
        """

        Parameters
        ----------
        item: Any
            the item to be searched in the names and coordinates.

        Returns
        -------
            a tuple to be interpreted as item(s) selector
        """
        txt = "item must be a tuple of len = 2 with the first element being the rows selection and"
        txt += " the second element the column selection."
        assert len(item) == 2, txt
        rows_item, cols_item = item

        def check_items(items, max_val):
            out = []
            for itm in items:
                if isinstance(itm, slice):
                    line = np.arange(
                        itm.start if itm.start is not None else 0,
                        itm.stop if itm.stop is not None else max_val,
                        itm.step if itm.step is not None else 1,
                    )
                elif isinstance(itm, np.ndarray):
                    line = itm.flatten().tolist()
                elif isinstance(itm, (int, float)):
                    line = [itm]
                else:
                    line = list(itm)
                out += [line]
            return np.array(out).flatten().tolist()

        # get the rows and cols indices
        if not isinstance(rows_item, (tuple, list, np.ndarray)):
            rows_item = [rows_item]
        rows = check_items(rows_item, self.coordinates.shape[0])
        if not isinstance(cols_item, (tuple, list, np.ndarray)):
            cols_item = [cols_item]
        cols = check_items(cols_item, self.coordinates.shape[1])

        # adjust the cols by the (existing) names attribute
        for i, v in enumerate(cols):
            try:
                cols[i] = np.int(v)
            except ValueError:
                ref = np.argwhere(self.names == str(v)).flatten()
                if len(ref) > 0:
                    cols[i] = ref[0]

        # get the validity of the selection
        out_of_range_rows = np.argwhere(
            rows not in np.arange(self.coordinates.shape[0])
        ).flatten()
        assert len(out_of_range_rows) == 0, "rows out of samples: {}".format(
            out_of_range_rows
        )
        out_of_range_cols = [np.isreal(i) for i in cols]
        assert np.all(out_of_range_cols), "cols out of samples: {}".format(
            out_of_range_cols
        )
        return rows, cols

    def __getitem__(self, item):
        """
        get a coordinate by name

        Parameters
        ----------
        item: str
            the name of the coordinate

        Returns
        -------
        val:    array-like
            the item required
        """
        vec = self.copy()
        keys = self._get_selection(item)
        vec.index = vec.index[keys[0]]
        vec.names = vec.names[keys[1]]
        vec.coordinates = vec.coordinates[keys]
        return vec

    def __setitem__(self, item, value):
        """
        set a new value to an item

        Parameters
        ----------
        item: numeric or str and array-like
            the list or atomic value representing the key(s) of the items to be set.

        value: array-like or int/float
            the value to be set.
        """
        keys = self._get_selection(item)
        if isinstance(value, (int, float)):
            val = np.vstack([np.tile(value, len(keys[1])) for i in keys[0]])
            self.coordinates[keys] = val
        elif isinstance(value, Vector):
            self.index[keys[0]] = value.index
            self.names[keys[1]] = value.names
            self.coordinates[keys] = value.coordinates
        elif isinstance(value, np.ndarray):
            shapes = [i == 1 for i in value.shape]
            if (np.any(shapes) and len(shapes) <= 2) or len(shapes) == 1:
                self.coordinates[keys] = value.flatten()
            elif value.ndim == 2:
                self.coordinates[keys[0]][:, keys[1]] = value
            else:
                txt = "only 1D or 2D arrays can be passed to __setitem__."
                raise NotImplementedError(txt)
        elif isinstance(value, (pd.DataFrame, pd.Series)):
            if np.any([i == 1 for i in value.shape]):
                self.coordinates[keys] = value.values.flatten()
            else:
                self.coordinates[keys[0]][:, keys[1]] = value.values
        else:
            txt = "the __setitem__ method does not currently support objects "
            txt += "of instance {}".format(isinstance(value))
            raise NotImplementedError(txt)

    def __getattr__(self, item):
        """
        get an attribute by name

        Parameters
        ----------
        item: str
            a coordinate dimension

        Returns
        -------
        dim: array-like
            return the coordinate corresponding to the required dimension
        """
        if hasattr(self, "names"):
            if item in self.names:
                return self.coordinates[:, np.argwhere(item == self.names).flatten()]
        else:
            raise AttributeError("{} not found.".format(item))

    def __setattr__(self, key, value):
        """
        set an attribute

        Parameters
        ----------
        key: str
            the name of the attribute

        value: Any
            the value to be set for the given attribute
        """
        if key in np.array(["coordinates", "names", "unit", "index"]):
            self.__dict__[key] = value
        elif key in self.names:
            try:
                self.coordinates[:, np.argwhere(key == self.names).flatten()] = value
            except Exception:
                raise ValueError("coordinate {} cannot be set to {}".format(key, value))
        else:
            self.names = np.append(self.names, [key])
            self.coordinates = np.hstack([self.coordinates, np.atleast_2d(value)])

    def __str__(self):
        """
        printing function
        """
        return self.to_df().__str__()

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
        return float(1 / np.mean(np.diff(self.index)))

    def to_df(self):
        """
        return a pandas DataFrame representing the object
        """
        return pd.DataFrame(
            data=self.coordinates,
            index=self.index,
            columns=[i + " ({})".format(self.unit) for i in self.names],
        )

    @property
    def shape(self):
        return (len(self.index), len(self.names))

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
        if (
            not np.sum(
                [
                    i - j
                    for i, j in zip(vector.coordinates.shape, self.coordinates.shape)
                ]
            )
            == 0
        ):
            return False
        if not np.all([i in vector.names for i in self.names]):
            return False
        if not np.all([i in vector.index for i in self.index]):
            return False
        if self.unit != vector.unit:
            return False
        return True

    def copy(self):
        """
        Return a copy of self.
        """
        return Vector(
            coordinates=self.coordinates,
            index=self.index,
            names=self.names,
            unit=self.unit,
        )

    def __add__(self, value):
        """
        The "self + value" operator.

        Parameters
        ----------
        value: Marker, array-like, float, int
            the value to be added to self.

        Returns
        -------
        out: Vector
            self with value being added to the coordinates.
        """
        vec = self.copy()
        return vec.__iadd__(value)

    def __radd__(self, value):
        """
        The "value + self" operator.

        Parameters
        ----------
        value: Marker, array-like, float, int
            the value to be added to self.

        Returns
        -------
        out: Vector
            self with value being added to the coordinates.
        """
        return self.__add__(value)

    def __iadd__(self, value):
        """
        The "self += value" operator.

        Parameters
        ----------
        value: Vector, array-like, float, int
            the value to be added to self.
        """
        if isinstance(value, Vector):
            assert self._matches(value)
            self.coordinates += value.coordinates
        else:
            self.coordinates += value
        return self

    def __sub__(self, value):
        """
        The "self - value" operator.

        Parameters
        ----------
        value: Marker, array-like, float, int
            the value to be added to self.

        Returns
        -------
        out: Vector
            self with value being removed to the coordinates.
        """
        vec = self.copy()
        return vec.__isub__(value)

    def __isub__(self, value):
        """
        The "self -= value" operator.

        Parameters
        ----------
        value: Vector, array-like, float, int
            the value to be subtracted to self.
        """
        if isinstance(value, Vector):
            assert self._matches(value)
            self.coordinates = self.coordinates - value.coordinates
        else:
            self.coordinates = self.coordinates - value
        return self

    def __mul__(self, value):
        """
        The "self * value" operator.

        Parameters
        ----------
        value: Marker, array-like, float, int
            the value to be added to self.

        Returns
        -------
        out: Vector
            self with value times the coordinates.
        """
        vec = self.copy()
        return vec.__imul__(value)

    def __rmul__(self, value):
        """
        The "value * self" operator.

        Parameters
        ----------
        value: Marker, array-like, float, int
            the value to be added to self.

        Returns
        -------
        out: Vector
            self with value times the coordinates.
        """
        return self.__mul__(value)

    def __imul__(self, value):
        """
        The "self *= value" operator.

        Parameters
        ----------
        value: Vector, array-like, float, int
            the value to be multiplied with self.
        """
        if isinstance(value, Vector):
            assert self._matches(value)
            self.coordinates *= value.coordinates
        else:
            self.coordinates *= value
        return self

    def __truediv__(self, value):
        """
        The "self / value" operator.

        Parameters
        ----------
        value: Marker, array-like, float, int
            the value to divide self by.

        Returns
        -------
        out: Vector
            self with the coordinates divided by the value.
        """
        vec = self.copy()
        return vec.__itruediv__(value)

    def __itruediv__(self, value):
        """
        The "self /= value" operator.

        Parameters
        ----------
        value: Vector, array-like, float, int
            the value to divide self by.
        """
        if isinstance(value, Vector):
            assert self._matches(value)
            self.coordinates /= value.coordinates
        else:
            self.coordinates /= value
        return self

    def __neg__(self):
        """
        Get the negative of the coordinates.
        """
        vec = self.copy()
        vec.coordinates *= -1
        return vec

    def __pow__(self, value):
        """
        Get the self ** value operation.

        Parameters
        ----------
        value: int, float
            the value be used as exponent for each element of the coordinates.

        Returns
        -------
        out: Vector
            a Marker with each coordinate elevated to value.
        """
        txt = "power operator is allowed only for float or int exponents."
        assert isinstance(value, (float, int)), txt
        vec = self.copy()
        vec.coordinates = vec.coordinates ** value
        return vec

    def __abs__(self):
        """
        Take the absolute values of all the coordinates.
        """
        vec = self.copy()
        vec.coordinates = abs(vec.coordinates)
        return vec

    def __eq__(self, other):
        """
        Equality compare.

        Parameters
        ----------
        other: Any
            any object to be compared with

        Returns
        -------
        matches: bool
            the result of the equality comparison
        """
        if self._matches(other):
            return self.to_df() == other.to_df()
        return False

    def __ne__(self, other):
        """
        Disequality compare.

        Parameters
        ----------
        other: Any
            any object to be compared with

        Returns
        -------
        matches: bool
            the result of the equality comparison
        """
        return not self.__eq__(other)

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
            coordinates=np.sqrt(np.sum(self.coordinates ** 2, axis=1)),
            names="||{}||".format("+".join(self.names)),
            index=self.index,
            unit=self.unit,
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
        to_excel(
            path=path, df=self.to_df(), sheet=sheet, keep_index=True, new_file=new_file
        )
