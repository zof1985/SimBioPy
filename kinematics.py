# IMPORTS

from base import *


# METHODS

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
    assert os.path.exists(path), path + ' does not exist.'
    assert path[-4:] == '.csv' or path[-4:] == '.txt', path + ' must be a ".csv" or ".txt" file.'

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
    assert os.path.exists(path), path + ' does not exist.'
    assert path[-5:] == '.xlsx' or path[-4:] == '.xls', path + ' must be a ".xlsx" or ".xls" file.'

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
    assert os.path.exists(path), path + ' does not exist.'
    assert path[-4:] == '.emt', path + ' must be an ".emt" path.'

    # read the path
    try:
        path = open(path, 'r')
        lines = [[j.strip() for j in i] for i in [i.split('\t') for i in path]]
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
    names = np.unique([i.split('.')[0] for i in vars[2:] if len(i) > 0])

    # get the data values
    values = np.vstack([np.atleast_2d(i[:len(vars)]) for i in lines[11:-2]]).astype(float)

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
        assert isinstance(origin, (list, np.ndarray)), "origin must be a list or numpy array."
        self.origin = np.atleast_1d(origin).flatten()

        # handle the names parameter
        if names is None:
            names = ["X{}".format(i + 1) for i in range(len(self.origin))]
        assert isinstance(names, (list, np.ndarray)), "names must be None, list or a numpy array."
        self.names = np.atleast_1d(names).flatten()

        # handle the orientation parameter
        if orientation is None:
            orientation = np.eye(len(self.origin))
        assert isinstance(orientation, (list, np.ndarray)), "orientation must be a list or numpy array."
        self.orientation = np.atleast_2d([gram_schmidt(i) for i in np.atleast_2d(orientation)])

    def __str__(self):
        """
        generate a pandas.DataFrame representing the ReferenceFrame.
        """
        origin_df = pd.DataFrame(data=np.atleast_2d(self.origin), columns=self.names)
        orientation_df = pd.DataFrame(
            data=self.orientation,
            columns=["dim{}".format(i + 1) for i in range(len(self.origin))],
            index=self.names
        )
        return {'Origin': origin_df, 'Orientation': orientation_df}

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
        if np.all([i in vector.names for i in self.names]):
            return False
        return True

    def copy(self):
        """
        Return a copy of self.
        """
        return ReferenceFrame(origin=self.origin, orientation=self.orientation, names=self.names)

    def rotate(self, vector):
        """
        Rotate vector according to the current ReferenceFrame instance.

        Parameters
        ----------
        vector: Vector, Vector
            a Vector or Vector object.

        Returns
        -------
        vec: Vector, Vector
            a rotated Vector/Vector object.
        """

        # ensure vector can be converted
        assert self._matches(vector), "vector must be a Vector or Vector object."

        # rotate marker's coordinates
        vec = vector.copy()
        vec.coordinates -= np.vstack([np.atleast_2d(self.origin) for i in range(vec.coordinates.shape[0])])
        vec.coordinates = vec.coordinates.dot(self.orientation.T)
        return vec

    def invert(self, vector):
        """
        Rotate vector back from the current Reference Frame to its former one.

        Parameters
        ----------
        vector: Vector, Vector
            a Vector or Vector object.

        Returns
        -------
        vec: Vector, Vector
            a rotated Vector/Vector object.
        """

        # ensure vector can be converted
        assert self._matches(vector), "vector must be a Vector or Vector object."

        # rotate marker's coordinates
        vec = vector.copy()
        vec.coordinates = vec.coordinates.dot(self.orientation)
        vec.coordinates += np.vstack([np.atleast_2d(self.origin) for i in range(vec.coordinates.shape[0])])
        return vec


class Vector:
    """
    Create a Vector object instance.
    """

    def __init__(self, coordinates, names=None, index=None, sampling_frequency=1., unit=""):
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
            assert isinstance(names, (list, np.ndarray)), "names must be a 1D array-like object."
            names = np.atleast_1d(names).flatten()
            if index is None:
                assert isinstance(sampling_frequency, (int, float)), "sampling_frequency must be numeric."
                index = np.linspace(0, sampling_frequency * coordinates.shape[0], coordinates.shape[0])
            else:
                txt = "index must be a 1D array-like object of len = {}".format(coordinates.shape[0])
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

    def _get_indices(self, item):
        """

        Parameters
        ----------
        item: Any
            the item to be searched in the names and coordinates.

        Returns
        -------
            a tuple to be interpreted as item(s) selector
        """
        if isinstance(item, tuple):
            item = list(item)
        elif not isinstance(item, (list, np.ndarray)):
            item = [item]
        any_slice = False
        for i, v in enumerate(item):
            if isinstance(v, slice):
                any_slice = True
            else:
                if not isinstance(v, (list, np.ndarray)):
                    v = [v]
                for j, k in enumerate(v):
                    if np.any(self.names == str(k)):
                        v[j] = np.argwhere(self.names == str(k)).flatten()[0]
                item[i] = v
        if not any_slice:
            item = [np.s_[:]] + np.atleast_2d(item).tolist()
        return tuple(item)

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
        return self.coordinates.__getitem__(self._get_indices(item))

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
        key = self._get_indices(item)
        if not isinstance(key, (list, tuple, np.ndarray)):
            key = [key]
            value = [value]
        for i, k in enumerate(key):
            index = np.argwhere(k == self.names).flatten()
            if len(index) == 0:
                self.names = np.append(self.names, [k])
                index = len(self.names) - 1
            self.coordinates[:, index] = value[i]


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
        return float(np.mean(np.diff(self.index)))


    def to_df(self):
        """
        return a pandas DataFrame representing the object
        """
        return pd.DataFrame(
            data=self.coordinates,
            index=self.index,
            columns=[i + " ({})".format(self.unit) for i in self.names]
        )


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
        if not np.sum([np.sum([i, j]) for i, j in zip(vector.coordinates.shape, self.coordinates.shape)]) == 0:
            return False
        if np.all([i in vector.names for i in self.names]):
            return False
        if np.all([i in vector.index for i in self.index]):
            return False
        if self.unit != vector.unit:
            return False
        return True


    def copy(self):
        """
        Return a copy of self.
        """
        return Vector(coordinates=self.coordinates, index=self.index, names=self.names, unit=self.unit)


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
        return self.copy().__iadd__(value)


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
        return self.copy().__isub__(value)


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
            self.coordinates -= value.coordinates
        else:
            self.coordinates -= value


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
        return self.copy().__imul__(value)


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
        return self.copy().__itruediv__(value)


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


    def __neg__(self):
        """
        Get the negative of the coordinates.
        """
        vec = self.copy()
        vec.coordinates *= (-1)
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
        txt = 'cross operator is valid only between 3D Vectors.'
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
            unit=self.unit
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
        to_excel(path=path, df=self.to_df(), sheet=sheet, keep_index=True, new_file=new_file)


class Container(dict):
    """
    Create a dict of or" object(s). It is a simple wrapper of the
    "dict" class with additional methods.

    Input:

        args (objects)

            objects of class "Point", "Point", or any subclass.
    """

    def to_csv(self, path, **kwargs):
        """
        store pandas.DataFrames containing the points formatted as:
        "XXX|YYY_ZZZ". where:
            'XXX' is the type of the vector
            'YYY' is the dimension of the vector
            'ZZZ'  if the dim_unit.

        In addition, the first column will be the index of the vectors.
        """

        # remove the filename from kwargs
        try:
            kwargs['path'] = None
        except Exception:
            pass

        # store all Vectors
        for v in self.keys():
            self[v].to_csv(os.path.sep.join([path, v + ".csv"]), **kwargs)

    def to_excel(self, path, new_file=False):
        """
        store an excel path containing the vectors formatted as: "XXX|YYY_ZZZ".

        where:
            'XXX' is the type of the vector
            'YYY' is the dimension of the vector
            'ZZZ'  if the dim_unit.

        In addition, the first column will be the index of the vectors.
        """

        # check if a new path must be created
        if new_file:
            os.remove(path)

        # store all Vectors
        [self[v].to_excel(path, v) for v in self]

    # GETTERS

    @staticmethod
    def from_csv(path, **kwargs):
        """
        Create a "VectorDict" object from a "csv" or "txt" path.

        Input:
            path: (str)
                an existing ".csv" or "txt" path or a folder containing csv
                files. The files must contain 1 column named "Index_ZZZ" and the
                others as "WWW:XXX|YYY_ZZZ" where:
                    'WWW' is the type of the vector
                    'XXX' is the name of the vector
                    'YYY' is the dimension of the vector
                    'ZZZ'  if the dim_unit.
        """

        # control the kwargs
        try:
            kwargs['index_col'] = False
        except Exception:
            pass

        # get the output dict
        vd = Container()

        # check if the path is a path or a folder and populate the
        # Container accordingly
        if os.path.isfile(path):
            vd[".".join(path.split(os.path.sep)[-1].split(".")[:-1])
            ] = _UnitDataFrame.from_csv(path, **kwargs)
        else:
            for i in get_files(path, ".csv", False):
                key = ".".join(i.split(os.path.sep)[-1].split(".")[:-1])
                vd[key] = _UnitDataFrame.from_csv(i, **kwargs)

        # return the dict
        return vd

    @staticmethod
    def from_excel(path, sheets=None, exclude_errors=True):
        """
        Create a "VectorDict" object from an excel path.

        Input:
            path:           (str)
                            an existing excel path. The sheets must contain 1
                            column named "Index_ZZZ" and the
                            others as "WWW:XXX|YYY_ZZZ" where:
                                'WWW' is the type of the vector
                                'XXX' is the name of the vector
                                'YYY' is the dimension of the vector
                                'ZZZ'  if the dim_unit.

            sheets:         (str, list or None)
                            the sheets to be imported. In None, all sheets are
                            imported.

            exclude_errors: (bool)
                            If a sheet generates an error during the import
                            would you like to skip it and import the
                            others?

        Output:
            a new VectorDict with the imported vectors.
        """

        vd = Container()

        # get the sheets
        dfs = from_excel(path, sheets)

        # import the sheets
        for i in dfs:
            if exclude_errors:
                try:
                    vd[i] = _UnitDataFrame.from_df(dfs[i])
                except Exception:
                    pass
            else:
                vd[i] = _UnitDataFrame.from_df(dfs[i])

        # return the dict
        return vd

    @staticmethod
    def from_emt(file):
        """
        Create a "VectorDict" object from a ".emt" path.

        Input:
            path: (str)
                an existing ".emt" path.
        """

        # check the validity of the entered path
        assert os.path.exists(file), file + ' does not exist.'
        assert file[-4:] == '.emt', file + ' must be an ".emt" path.'

        # read the path
        try:
            file = open(file)

            # get the lines of the path
            lines = [[j.strip() for j in i]
                     for i in [i.split('\t') for i in file]]

        # something went wrong so close path
        except Exception:
            lines = []

        # close the path
        finally:
            file.close()

        # get the output VectorDict
        vd = _UnitDataFrame()

        # get the units
        dim_unit = lines[3][1]
        time_unit = 's'

        # get the type
        type = lines[2][1]

        # get an array with all the variables
        V = np.array([i for i in lines[10] if i != ""]).flatten()

        # get the data names
        names = np.unique([i.split('.')[0] for i in V[2:] if len(i) > 0])

        # get the data values (now should work)
        values = np.vstack(
            [np.atleast_2d(i[:len(V)])
             for i in lines[11:-2]]
        ).astype(float)

        # get the columns of interest
        cols = np.arange(np.argwhere(V == "Time").flatten()[0] + 1, len(V))

        # get the rows in the data to be extracted
        rows = np.argwhere(np.any(~np.isnan(values[:, cols]), 1)).flatten()
        rows = np.arange(np.min(rows), np.max(rows) + 1)

        # get time
        time = values[rows, 1].flatten()

        # generate a dataframe for each variable
        for v in names:

            # get the dimensions
            D = [i.split(".")[-1] for i in V if i.split(".")[0] == v]
            D = [""] if len(D) == 1 else D

            # get the data for each dimension
            K = {}
            for i in D:
                key = i if i != "" else v
                cols = np.argwhere(V == v + (("." + i) if i != "" else ""))
                K[key] = values[rows, cols.flatten()]

            # setup the output variable
            vd[v] = _UnitDataFrame(
                data=K,
                index=time,
                time_unit=time_unit,
                dim_unit=dim_unit,
                type=type
            )

        # return vd
        return vd

    # SUBCLASSED METHODS

    def __init__(self, *args, **kwargs):
        super(Container, self).__init__(*args, **kwargs)
        self.__finalize__()

    def __finalize__(self):
        for i in self.keys():
            assert isinstance(
                self[i], _UnitDataFrame
            ), "{} is not a Point".format(i)
        return self

    def __str__(self):
        lst = []
        for i in self.keys():
            lst += [" ".join(["\n\nVector:\t ", i, "\n\n", self[i].__str__()])]
        return "\n".join(lst)

    def __repr__(self):
        return self.__str__()

    def __setitem__(self, *args, **kwargs):
        super(Container, self).__setitem__(*args, **kwargs)
        self.__finalize__()

    def __setattr__(self, *args, **kwargs):
        super(Container, self).__setattr__(*args, **kwargs)
        self.__finalize__()
