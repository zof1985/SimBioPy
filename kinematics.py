
# IMPORTS

from base import *
import os
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation


# CLASSES

class ReferenceFrame():
    """
    Create a ReferenceFrame instance.

    Input

        origin (numpy 1 or 2 dimensional array)

            a numpy array containing the origin of the "local"
            reference frame. 'origin' is casted to a 2D ndarray and it must
            result in 1 single row with n-dimensions.

        orientation (numpy 2 dimensional array)

            a 2D numpy array containing the unit vectors defining the
            orientation of the ReferenceFrame with respect to the "global"
            Frame.
            The orientation must be a 2D numpy array where each row is a versor
            and each column a dimension.

        names (list, numpy 1 dimensional array)

            the names of each dimension provided as array or list of strings.

        unit (str)

            the unit of measurement of the ReferenceFrame.
    """

    def __init__(
        self,
        origin = np.array([0, 0, 0]),
        orientation = np.array([[1, 0, 0],
                                [0, 1, 0],
                                [0, 0, 1]]),
        names = np.array(['X', 'Y', 'Z']),
        unit = "m"
        ):

        # check the names
        txt = "'{}' must be a numpy array or list."
        valid = (list, np.ndarray)
        assert isinstance(names, valid), txt.format("names")
        N = np.array([names]).flatten()
        self.ndim = len(N)

        # handle the origin
        assert isinstance(origin, valid), txt.format("origin")
        O = np.array([origin]).flatten().astype(np.float)
        assert len(O) == self.ndim, "'origin' len must be {}".format(self.ndim)
        self.origin = pd.DataFrame(O, columns = ['Origin'], index = N).T

        # handle the unit
        assert isinstance(unit, (str)), "'unit' must be a str."
        self.unit = unit

        # handle the orientation
        assert isinstance(orientation, valid), txt.format("orientation")
        O = np.atleast_2d(orientation)
        r, c = O.shape
        txt = "'orientation' must be a {} x {} matrix."
        txt = txt.format(self.ndim, self.ndim)
        assert r == c == self.ndim, txt
        self.versors = pd.DataFrame(
            data    = self._orthogonalize(O),
            index   = ['v{}'.format(i+1) for i in np.arange(self.ndim)],
            columns = N
            )

        # get the dimension names
        self.dim = self.versors.columns.to_list()

        # calculate the rotation matrix
        self._rot = Rotation.from_matrix(self.versors)

    def _orthogonalize(self, vectors):
        """
        return the orthogonal basis defined by a set of vectors using the
        Gram-Schmidt algorithm.

        Input

            normalized (bool)

                should the projected points returned in normalized units?

            args / kwargs (Point)

                one or more points from which the orthogonal projections have
                to be calculated.

        Output

            W (Container)

                a Container object containing the orthogonal points.
        """

        # internal functions to simplify the calculation
        def proj(a, b):
            return (np.inner(a, b) / np.inner(b, b) * b).astype(np.float)

        def norm(v):
            return v / np.sqrt(np.sum(v ** 2))

        # calculate the projection points
        W = []
        for i, u in enumerate(vectors):
            w = np.copy(u).astype(np.float)
            for j in vectors[:i, :]:
                w -= proj(u, j)
            W += [w]

        # normalize
        return np.vstack([norm(u) for u in W])

    def __str__(self):
        D = self.to_dict()
        return D[[i for i in D][0]].__str__() + "\n"

    def __repr__(self):
        return self.__str__()

    def __eq__(self, other):
        if isinstance(other, (ReferenceFrame)):
            return (
                (other.ndim == self.ndim) &
                (other.nsamples == self.nsamples) &
                np.all([i in np.array(self.dim) for i in other.dim]) &
                (np.sum(self.origin.values - other.origin.values) == 0) &
                (np.sum(self.versors.values - other.versors.values) == 0) &
                (other.unit == self.unit)
            )
        return False

    def __ne__(self, other):
        return not self.__eq__(other)

    def to_dict(self):
        """
        return a dict representing the content of the ReferenceFrame.
        """
        O = pd.concat([self.origin, self.versors], axis = 0)
        C = O.columns.to_list()
        O.columns = pd.Index([(i + " ({})".format(self.unit)) for i in C])
        return {'ReferenceFrame': O}

    def copy(self):
        """
        return a copy of the object.
        """
        return ReferenceFrame(
            origin      = self.origin.values,
            orientation = self.versors.values,
            names       = self.dim,
            unit        = self.unit
            )


class Marker(ReferenceFrame):
    """
    Create a Marker object instance.

    Input

        coords (numpy 2 dimensional array, pandas.DataFrame)

            the coordinates of the marker within the provided ReferenceFrame.
            each row denotes one single sample.
            if a pandas.DataFrame is provided, the columns are used as
            names of the dimensions.

        fs (float)
            the sampling rate (in Hz) at which the coordinates are sampled.

        origin (numpy 1 or 2 dimensional array)

            a numpy array containing the origin of the "local"
            reference frame. 'origin' is casted to a 2D ndarray and it must
            result in 1 single row with n-dimensions.

        orientation (numpy 2 dimensional array)

            a 2D numpy array containing the unit vectors defining the
            orientation of the ReferenceFrame with respect to the "global"
            Frame.
            The orientation must be a 2D numpy array where each row is a versor
            and each column a dimension.

        names (list, numpy 1 dimensional array)

            the names of each dimension provided as array or list of strings.
            In case the

        unit (str)

            the unit of measurement of the ReferenceFrame.
    """

    def __init__(
        self,
        coords,
        fs = 1,
        origin  = np.array([0, 0, 0]),
        orientation = np.array([[1, 0, 0],
                               [0, 1, 0],
                               [0, 0, 1]]),
        names   = np.array(['X', 'Y', 'Z']),
        unit    = "m"
        ):

        # initialize the ReferenceFrame
        super().__init__(origin, orientation, names, unit)

        # check the sample frequency
        txt = "'fs' must be a positive float."
        assert isinstance(fs, (float)) and fs > 0, txt
        self.fs = fs

        # check the coordinates
        txt = "coords must be a 2D numpy array or a pandas DataFrame with "
        txt += "{} dimensions.".format(self.ndim)
        assert isinstance(coords, (pd.DataFrame, np.ndarray)), txt
        txt = "'coords' must have {} dimensions.".format(self.ndim)
        if isinstance(coords, (pd.DataFrame)):
            N = coords.columns
            assert len(N.to_list()) == self.ndim, txt
            self.origin.columns = N
            self.versors.columns = N
            C = coords.values
        else:
            N = self.origin.columns
            C = np.atleast_2d(coords)
            assert C.shape[1] == self.ndim, txt

        # get the coordinates
        self.coordinates = pd.DataFrame(
            data    = C,
            index   = np.arange(C.shape[0]) * fs,
            columns = N
            )

        # set the number of samples
        self.nsamples = self.coordinates.shape[0]

    def __str__(self):
        D = self.to_dict()
        O = ""
        for d in D:
            O += d + ":\n" + D[d].__str__() + "\n\n"
        return O

    def _validate_arg(self, value):
        """
        private method used to ensure that value can be used as argument for
        arithmetic operators.

        Input

            value (Marker, Vector, pandas.DataFrame, numpy.ndarray, float, int)

        Output

            array (numpy.ndarray)

                a 2D numpy array ready to be used for arithmetic operations
                with the Marker coordinates.
        """

        # handle Marker
        if isinstance(value, (Marker)):

            # align the right hand element of the sum to the ReferenceFrame
            # of the left hand side of the equation
            V = value.change_frame(self)

            # ensure that the coordinates sum is possible
            idx = V.coordinates.index.to_numpy()
            iii = self.coordinates.index.to_numpy()
            txt = "coordinates index does not match between the arguments."
            assert np.all([i in iii for i in idx]), txt

            # get the array
            A = V.coordinates.values

        # handle dataframes
        elif isinstance(value, (pd.DataFrame)):

            # ensure the coordinates can be added together
            D = np.array(self.dim).flatten()
            txt = "'value' must have columns: {}".format(D)
            assert np.all([i in D for i in value.columns.to_list()]), txt

            # get the array
            A = value.values

        # handle numpy arrays
        elif isinstance(value, (np.ndarray, list)):

            # ensure value has an applicable format
            A = np.atleast_2d(value)
            if A.shape[1] == 1:
                A = np.hstack([A for i in np.arange(self.ndim)])
            elif A.shape[0] == 1:
                A = np.vstack([A for i in np.arange(self.nsamples)])
            txt = "value's shape is not compatible with '+' operator."
            assert A.shape[0] == self.nsamples and A.shape[1] == self.ndim, txt

        # handle scalar values
        elif isinstance(value, (float, int)):

            # make an array replicating the value
            A = np.atleast_2d([value])
            A = np.hstack([A for i in np.arange(self.ndim)])
            A = np.vstack([A for i in np.arange(self.nsamples)])

        # objects of other instances are not supported
        else:
            TypeError("'value' type not supported for this operation.")

        # return the array
        return A

    def __add__(self, value):
        V = self.copy()
        V.coordinates += self._validate_arg(value)
        return V

    def __radd__(self, value):
        return self.__add__(value)

    def __iadd__(self, value):
        self.coordinates += self._validate_arg(value)

    def __sub__(self, value):
        V = self.copy()
        V.coordinates -= self._validate_arg(value)
        return V

    def __rsub__(self, value):
        return self.__sub__(value)

    def __isub__(self, value):
        self.coordinates -= self._validate_arg(value)

    def __mul__(self, value):
        not_valid = (Marker)
        if isinstance(value, not_valid):
            txt = "scalar product is not a valid operation between {} objects."
            TypeError(txt.format(not_valid))
        V = self.copy()
        V.coordinates *= self._validate_arg(value)
        return V

    def __rmul__(self, value):
        return self.__mul__(value)

    def __imul__(self, value):
        not_valid = (Marker)
        if isinstance(value, not_valid):
            txt = "scalar product is not a valid operation between {} objects."
            TypeError(txt.format(not_valid))
        self.coordinates *= self._validate_arg(value)

    def __truediv__(self, value):
        not_valid = (Marker)
        if isinstance(value, not_valid):
            txt = "true division is not a valid operation between {} objects."
            TypeError(txt.format(not_valid))
        V = self.copy()
        V.coordinates /= self._validate_arg(value)
        return V

    def __rtruediv__(self, value):
        return self.__truediv__(value)

    def __itruediv__(self, value):
        not_valid = (Marker)
        if isinstance(value, not_valid):
            txt = "true division is not a valid operation between {} objects."
            TypeError(txt.format(not_valid))
        self.coordinates /= self._validate_arg(value)

    def __neg__(self):
        V = self.copy()
        V.coordinates *= (-1)
        return V

    def __pow__(self, value):
        not_valid = (Marker, np.ndarray, pd.DataFrame)
        if isinstance(value, not_valid):
            txt = "{} objects cannot be used as power arguments."
            TypeError(txt.format(not_valid))
        V = self.copy()
        V.coordinates ** self._validate_arg(value)
        return V

    def __abs__(self):
        V = self.copy()
        V.coordinates = abs(V.coordinates)
        return V

    def __matmul__(self, value):
        if isinstance(value, (pd.DataFrame)):
            A = value.copy()
        elif isinstance(value, (np.ndarray)):
            A = pd.DataFrame(
                data = np.atleast_2d(value),
                index = ['N{}'.format(i) for i in np.arange(value.shape[0])],
                columns = ['X{}'.format(i) for i in np.arange(value.shape[1])]
            )
        elif isinstance(value, (float, int)):
            A = pd.DataFrame(
                data    = np.atleast_2d([value]),
                index   = ["N0"],
                columns = ["X0"]
            )
        else:
            TypeError("'value' type not supported for this operation.")

        # ensure the shape of value is compatible with matrix product
        r = A.shape[0]
        txt = "value must have {} rows".format(self.ndim)
        assert r == self.ndim, txt

        # perform the matrix multiplication
        A.index = self.coordinates.columns
        return self.coordinates.dot(A)

    def __rmatmul__(self, value):
        if isinstance(value, (pd.DataFrame)):
            A = value.copy()
        elif isinstance(value, (np.ndarray)):
            A = pd.DataFrame(
                data    = np.atleast_2d(value),
                index   = ['N{}'.format(i) for i in np.arange(value.shape[0])],
                columns = ['X{}'.format(i) for i in np.arange(value.shape[1])]
            )
        elif isinstance(value, (float, int)):
            A = pd.DataFrame(
                data    = np.atleast_2d([value]),
                index   = ["N0"],
                columns = ["X0"]
            )
        else:
            TypeError("'value' type not supported for this operation.")

        # ensure the shape of value is compatible with matrix product
        r = A.shape[1]
        txt = "value must have {} columns".format(self.nsamples)
        assert r == self.nsamples, txt

        # perform the matrix multiplication
        V = self.coordinates.copy()
        V.index = A.columns
        return A.dot(V)

    def __eq__(self, other):
        if isinstance(other, (Marker)):
            return (
                super().__eq__(other) &
                (np.sum((other.coordinates - self.coordinates).values) == 0) &
                (np.sum((other.coordinates.index.to_numpy() -
                         self.coordinates.index.to_numpy())) == 0)
            )
        return False

    def __ne__(self, other):
        return not self.__eq__(other)

    def change_frame(self, R):
        """
        view the current object from another Reference Frame.

        Input

            R (ReferenceFrame)

                the ReferenceFrame object on which the current Marker has to be
                aligned.

        Output

            M (Marker)

                the marker represented by the ReferenceFrame R
        """

        # check the reference frame
        assert isinstance(R, (ReferenceFrame)), "'R' must be a ReferenceFrame."
        assert self.unit == R.unit, "'R' unit diverge from coordinates unit."
        C = self.coordinates.columns.to_numpy()
        N = R.origin.columns.to_numpy()
        assert np.all([i in N for i in C]), "R dimensions must be {}".format(C)
        assert self.unit == R.unit, "'R' unit diverge from coordinates unit."

        # move the coordinates of the actual marker to the "global" frame
        M0 = self._rot.inv().apply(self.coordinates) + self.origin.values
        M1 = R._rot.apply(M0 - R.origin.values)
        return Marker(
            coords      = M1,
            fs          = self.fs,
            origin      = R.origin.values,
            orientation = R.versors.values,
            names       = self.coordinates.columns.to_list(),
            unit        = R.unit
            )

    def to_dict(self):
        """
        create a dict object summarizing all the relevant components of the
        Marker.

        Output:

            D (dict)

                a dict with 2 keys (Coordinates and ReferenceFrame) containing
                a pandas DataFrame with all the relevant data.
        """
        O = self.coordinates.copy()
        C = O.columns.to_list()
        O.columns = pd.Index([(i + " ({})".format(self.unit)) for i in C])
        O.index = pd.Index([(str(i) + " (s)") for i in O.index.to_numpy()])
        return {'Coordinates': O, **super().to_dict()}

    def copy(self):
        """
        return a copy of the object.
        """
        return Marker(
            coords      = self.coordinates.values,
            fs          = self.fs,
            origin      = self.origin.values,
            orientation = self.versors.values,
            names       = self.dim,
            unit        = self.unit
            )

    def as_vector(self):
        """
        return a copy of the current Marker as Vector instance.
        """
        return Vector(
            coords      = self.coordinates.values,
            fs          = self.fs,
            origin      = self.origin.values,
            orientation = self.versors.values,
            names       = self.dim,
            unit        = self.unit
            )

    def as_referenceframe(self):
        """
        return the ReferenceFrame instance of the current object.
        """
        return ReferenceFrame(
            origin      = self.origin.values,
            orientation = self.versors.values,
            names       = self.dim,
            unit        = self.unit
            )


class Vector(Marker):
    """
    Create a Vector object instance.

    Input

        coords (numpy 2 dimensional array, pandas.DataFrame)

            the coordinates of the marker within the provided ReferenceFrame.
            each row denotes one single sample.
            if a pandas.DataFrame is provided, the columns are used as
            names of the dimensions.

        fs (float)
            the sampling rate (in Hz) at which the coordinates are sampled.

        origin (numpy 1 or 2 dimensional array)

            a numpy array containing the origin of the "local"
            reference frame. 'origin' is casted to a 2D ndarray and it must
            result in 1 single row with n-dimensions.

        orientation (numpy 2 dimensional array)

            a 2D numpy array containing the unit vectors defining the
            orientation of the ReferenceFrame with respect to the "global"
            Frame.
            The orientation must be a 2D numpy array where each row is a versor
            and each column a dimension.

        names (list, numpy 1 dimensional array)

            the names of each dimension provided as array or list of strings.
            In case the

        unit (str)

            the unit of measurement of the ReferenceFrame.
    """
    def __init__(
        self,
        coords,
        fs          = 1,
        origin      = np.array([0, 0, 0]),
        orientation = np.array([[1, 0, 0],
                                [0, 1, 0],
                                [0, 0, 1]]),
        names       = np.array(['X', 'Y', 'Z']),
        unit        = "m"
        ):
        super().__init__(coords, fs, origin, orientation, names, unit)

    def cross(self, value):
        """
        get the cross product between 3D Vectors.

        Input:

            value (Vector)

                a 3D vector to be cross multiplied to self.
        """
        # check value
        return(Vector(
            coords      = np.cross(self.coordinates, self._validate_arg(value)),
            fs          = self.fs,
            origin      = self.origin.values,
            orientation = self.versors.values,
            names       = self.dim,
            unit        = self.unit
            ))

    def norm(self):
        """
        get the norm of the vector.

        Output

            N (pandas.DataFrame)

                return a DataFrame with one column containing the norm of the
                vector.
        """
        return pd.DataFrame(
            data    = np.sqrt(np.sum(self.coordinates ** 2, 1)),
            columns = ["||" + "+".join(self.dim) + "||"],
            index   = self.coordinates.index
            )

    def rotation_from(self, B):
        """
        obtain the scipy.spatial_transform.Rotation class object describing
        the rotation from B to the current vector:

        Input

            B (Vector)

                the vector from which the angle has to be calculated.
                If B has a different ReferenceFrame than self, it is firstly
                aligned to it.

        Output

            R (scipy.spatial_transform.Rotation)

                The rotation to be applied for obtaining self from B.
        """

        # check the entries
        assert isinstance(B, (Vector)), "'B' must be a Vector object."
        txt = "'B' must have dimensions {}.".format(self.dim)
        assert B.ndim == self.ndim, txt
        assert np.all([i in np.array(self.dim) for i in B.dim]), txt
        txt = "'B' must have {} samples.".format(self.nsamples)
        assert B.nsamples == self.nsamples, txt
        txt = "'B' samples does not match with self.coordinates.index."
        ii = self.coordinates.index.to_numpy()
        assert np.all([i in ii for i in B.coordinates.index.to_numpy()]), txt

        # obtain the vector (w) around which the rotation is provided
        w = B.cross(self)
        w = w / w.norm().values

        # get the rotation angle
        C = (B - self).norm()
        A = self.norm()
        K = B.norm()
        zeros = np.argwhere((C.values == 0) | (A.values == 0) | (K.values == 0))
        f = np.arccos((A ** 2 + K ** 2 - C ** 2) / (2 * A * K)).values
        f[zeros] = 0

        # return the rotations
        return Rotation.from_rotvec(w.coordinates.values * f)

    def rotation_to(self, B):
        """
        obtain the scipy.spatial_transform.Rotation class object describing
        the rotation from the current vector to B:

        Input

            B (Vector)

                the vector from which the angle has to be calculated.
                If B has a different ReferenceFrame than self, it is firstly
                aligned to it.

        Output

            R (scipy.spatial_transform.Rotation)

                The rotation to be applied for obtaining B from self.
        """
        return self.rotation_from(B).inv()

    def copy(self):
        """
        return a copy of the object.
        """
        return Vector(
            coords      = self.coordinates.values,
            fs          = self.fs,
            origin      = self.origin.values,
            orientation = self.versors.values,
            names       = self.dim,
            unit        = self.unit
            )

    def as_marker(self):
        """
        return a copy of the current object as Marker instance.
        """
        return Marker(
            coords      = self.coordinates.values,
            fs          = self.fs,
            origin      = self.origin.values,
            orientation = self.versors.values,
            names       = self.dim,
            unit        = self.unit
            )


class Container(dict):
    """
    Create a dict of or" object(s). It is a simple wrapper of the
    "dict" class with additional methods.

    Input:

        args (objects)

            objects of class "Point", "Point", or any subclass.
    """

    # STORING METHODS

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
        store an excel file containing the vectors formatted as: "XXX|YYY_ZZZ".

        where:
            'XXX' is the type of the vector
            'YYY' is the dimension of the vector
            'ZZZ'  if the dim_unit.

        In addition, the first column will be the index of the vectors.
        """

        # check if a new file must be created
        if new_file:
            os.remove(path)

        # store all Vectors
        [self[v].to_excel(path, v) for v in self]

    # GETTERS

    @staticmethod
    def from_csv(path, **kwargs):
        """
        Create a "VectorDict" object from a "csv" or "txt" file.

        Input:
            path: (str)
                an existing ".csv" or "txt" file or a folder containing csv
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

        # check if the path is a file or a folder and populate the
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
        Create a "VectorDict" object from an excel file.

        Input:
            path:           (str)
                            an existing excel file. The sheets must contain 1
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
        Create a "VectorDict" object from a ".emt" file.

        Input:
            file: (str)
                an existing ".emt" file.
        """

        # check the validity of the entered file
        assert os.path.exists(file), file + ' does not exist.'
        assert file[-4:] == '.emt', file + ' must be an ".emt" file.'

        # read the file
        try:
            file = open(file)

            # get the lines of the file
            lines = [[j.strip() for j in i]
                     for i in [i.split('\t') for i in file]]

        # something went wrong so close file
        except Exception:
            lines = []

        # close the file
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
        values = np.vstack([np.atleast_2d(i[:len(V)])
                            for i in lines[11:-2]]).astype(float)

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
                self[i], _UnitDataFrame), "{} is not a Point".format(i)
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


def match(*args, **kwargs):
    """
    check if the entered objects are comparable

    Output

        C (bool)

            True if all inputs are Points or a pandas.DataFrame with the
            same columns and index of self. False, otherwise.
    """

    # get the elements entered
    objs = [i for i in args] + [kwargs[i] for i in kwargs]

    # check the columns and index of all objs
    IX = objs[0].index.to_numpy()
    CL = objs[0].columns.to_numpy()
    SH = objs[0].shape
    TY = objs[0].__class__
    for obj in objs:
        OI = obj.index.to_numpy()
        OC = obj.columns.to_numpy()
        col_check = np.all([i in OC for i in CL])
        idx_check = np.all([i in OI for i in IX])
        shp_check = np.all([i == j for i, j in zip(obj.shape, SH)])
        cls_check = isinstance(obj, (TY))
        if not np.all([col_check, idx_check, shp_check, cls_check]):
            return False
    return True


def angle_between_2_vectors(A, B):
    """
    return the angle between the vectors A and B originating from the same
    point using the Cosine theorem.

    Input

        A (Vector)

            The coordinates of one point.

        B (Vector)

            The coordinates of the point over which the angle has
            to be calculated.

    Output

        K (DF)
            A 1D point containing the result of:
                                 2          2            2
                       /  (A - B)  +  (C - B)  -  (A - C)  \
                arcos | ----------------------------------- |
                       \      2 * (A - B) * (C - B)        /
    """

    # ensure all entered parameters are points
    txt = "'A' and 'B' must be Points with equal index and columns."
    assert match(A, B), txt

    # get a, b and c
    a = A.norm()
    b = B.norm()
    c = (A - B).norm()

    # return the angle
    k = (a ** 2 + b ** 2 - c ** 2) / (2 * a * b).values
    k.loc[k.index] = np.arccos(k.values)
    k.time_unit = A.time_unit
    k.dim_unit = "rad"
    k.type = "Angle"
    return k

def angle_by_3_points(A, B, C):
        """
        return the angle ABC using the Cosine theorem.

        Input:
            A:  (Point)
                The coordinates of one point.

            B:  (Point)
                The coordinates of the point over which the angle has
                to be calculated.

            C:  (Point)
                The coordinates of the third point.

        Output:
            K:  (Point)
                A 1D point containing the result of:
                                     2          2            2
                           /  (A - B)  +  (C - B)  -  (A - C)  \
                    arcos | ----------------------------------- |
                           \      2 * (A - B) * (C - B)        /
        """

        # ensure all entered parameters are points
        txt = "'A', 'B' and 'C' must be Points with equal index and columns."
        assert _UnitDataFrame.match(A, B, C), txt

        # get a, b and c
        a = (A - B).norm()
        b = (C - B).norm()
        c = (A - C).norm()

        # return the angle
        k = (a ** 2 + b ** 2 - c ** 2) / (2 * a * b).values
        k.loc[k.index] = np.arccos(k.values)
        k.time_unit = A.time_unit
        k.dim_unit = "rad"
        k.type = "Angle"
        return k
