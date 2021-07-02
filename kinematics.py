# IMPORTS

import os

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation



def gram_schmidt(vectors):
    """
    return the orthogonal basis defined by a set of vectors using the
    Gram-Schmidt algorithm.

    Input

        vectors (DxD numpy.ndarray)

            a set of vectors with D dimensions to be orthogonalized

    Output

        W (DxD numpy.ndarray)

            the array containing the orthogonalized unit orientation.
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


class ReferenceFrame:
    """
    Create a ReferenceFrame instance.

    Input

        origin (numpy 2 dimensional array)

            a numpy array containing the origin of the "local"
            reference frame. 'origin' must have shape N x D where N are the
            number of samples and D the number of dimensions.


        orientation (numpy 3 dimensional array)

            a 3D numpy array with shape N x D x D, where N are the number of
            samples and D the number of dimensions. Please note that the 2nd
            dimension defines the orientation, while the 3rd the coordinates of each
            versor in the provided dimensions.

        sampling_frequency (float)

            the sampling frequency (in Hz) at which the samples have been
            collected.

        names (list, numpy 1 dimensional array)

            the names of each dimension provided as array or list of strings.

        unit (str)

            the unit of measurement of the ReferenceFrame.
    """


    def __init__(
            self,
            origin = np.array([[0, 0, 0]]),
            orientation = None,
            sampling_frequency = 1,
            names = None,
            unit = "m"
            ):

        # check the sampling frequency
        valid = (int, float)
        assert isinstance(sampling_frequency, valid), "'sampling_frequency' must be an int or float value."
        self.sampling_frequency = sampling_frequency

        # handle the origin parameter
        self.origin = np.atleast_2d(origin)
        self.sample_number, self.dimension_number = self.origin.shape

        # handle the names parameter
        if names is None:
            names = ["X{}".format(i + 1) for i in range(origin.shape[1])]
        self.dimensions = names

        # handle the orientation parameter
        if orientation is None:
            r, c = self.origin.shape
            orientation = np.concatenate([np.reshape(np.eye(c), [1, c, c]) for _ in np.arange(r)], axis = 0)
        assert isinstance(orientation, np.ndarray), "orientation must be a numpy 3D array."
        assert orientation.ndim == 3, "orientation must be a numpy 3D array"
        self.orientation = np.atleast_3d([gram_schmidt(i) for i in np.atleast_3d(orientation)])
        N, R, D = self.orientation.shape

        # get the time reference for the objects
        self.time = sampling_frequency * np.arange(N)

        # store everything
        txt = "'orientation' must be a {} x {} x {} matrix.".format(N, R, D)
        assert R == D == self.dimension_number, txt
        assert N == self.sample_number, txt

        # handle the unit
        assert isinstance(unit, str), "'unit' must be a str."
        self.unit = unit


    def to_df(self):
        """
        generate a pandas.DataFrame representing the ReferenceFrame.
        """
        cols = ["Origin {} ({})".format(d, self.unit) for d in self.dimensions]
        data = [self.origin]
        for i, d in enumerate(self.dimensions):
            data += [self.orientation[:, i, :]]
            cols += ["Versor {} {} ({})".format(i + 1, k, self.unit) for k in self.dimensions]
        data = np.hstack(data)
        index = pd.Index(self.time, name = "Time (s)")
        return pd.DataFrame(data = data, columns = cols, index = index)


    def _is_comparable_to_(self, B):
        """
        private method used to evaluate if B is similar to ReferenceFrame.

        Input

            B (ReferenceFrame)

                a ReferenceFrame object to be compared.

        Output

            C (bool)

                True if the dimensions, time index and unit are the same.
                False otherwise.
        """
        if not isinstance(B, (Marker, Vector, ReferenceFrame)):
            return False
        if self.sample_number != B.sample_number:
            return False
        if np.sum(self.time - B.time) != 0:
            return False
        if self.dimension_number != B.dimension_number:
            return False
        if self.unit != B.unit:
            return False
        if np.any([self.dimensions[i] != B.dimensions[i] for i in np.arange(self.dimension_number)]):
            return False
        return True


    def __str__(self):
        return self.to_df().__str__()


    def __repr__(self):
        return self.__str__()


    def __eq__(self, other):
        if isinstance(other, ReferenceFrame):
            return self.to_df() == other.to_df()
        return False


    def __ne__(self, other):
        return not self.__eq__(other)


    def copy(self):
        """
        return a copy of the object.
        """
        return ReferenceFrame(
                origin = self.origin,
                orientation = self.orientation,
                sampling_frequency = self.sampling_frequency,
                names = self.dimensions,
                unit = self.unit
                )


class Marker(ReferenceFrame):
    """
    Create a Marker object instance.

    Input

        coordinates (numpy 2 dimensional array, pandas.DataFrame)

            the coordinates of the marker within the provided ReferenceFrame.
            each row denotes one single sample.
            if a pandas.DataFrame is provided, the columns are used as
            names of the dimensions.

        origin (numpy 2 dimensional array)

            a numpy array containing the origin of the "local"
            reference frame. 'origin' must have shape N x D where N are the
            number of samples and D the number of dimensions.


        orientation (numpy 3 dimensional array)

            a 3D numpy array with shape N x D x D, where N are the number of
            samples and D the number of dimensions. Please note that the 2nd
            dimension defines the orientation, while the 3rd the coordinates of each
            versor in the provided dimensions.

        sampling_frequency (float)

            the sampling frequency (in Hz) at which the samples have been
            collected.

        names (list, numpy 1 dimensional array)

            the names of each dimension provided as array or list of strings.

        unit (str)

            the unit of measurement of the ReferenceFrame.
    """


    def __init__(
            self,
            coordinates = np.array([[0, 0, 0]]),
            origin = None,
            orientation = None,
            sampling_frequency = 1,
            names = None,
            unit = "m"
            ):

        # handle the coordinates parameter
        txt = "coordinates must be a 2D numpy array or a pandas DataFrame."
        assert isinstance(coordinates, (pd.DataFrame, np.ndarray)), txt
        if isinstance(coordinates, pd.DataFrame):
            names = coordinates.columns.to_numpy()
            coordinates = coordinates.values
        else:
            coordinates = np.atleast_2d(coordinates)

        # handle the names parameter
        if names is None:
            names = ["X{}".format(i + 1) for i in range(coordinates.shape[1])]

        # handle the origin parameter
        if origin is None:
            origin = np.zeros_like(coordinates)
        else:
            origin = np.atleast_2d(origin)

        # handle the orientation parameter
        if orientation is None:
            r, c = coordinates.shape
            orientation = np.concatenate([np.reshape(np.eye(c), [1, c, c]) for _ in np.arange(r)], axis = 0)

        # initialize the ReferenceFrame
        super().__init__(origin, orientation, sampling_frequency, names, unit)

        # check the coordinates
        txt = "coordinates must be a 2D numpy array or a pandas DataFrame with "
        txt += "{} rows and {} columns.".format(self.sample_number, self.dimension_number)
        assert isinstance(coordinates, (pd.DataFrame, np.ndarray)), txt
        N, D = coordinates.shape
        assert N == self.sample_number, txt
        assert D == self.dimension_number, txt
        if isinstance(coordinates, pd.DataFrame):
            self.dim = coordinates.columns.to_list()
            self.coordinates = coordinates.values
        else:
            self.coordinates = np.copy(coordinates)


    def _validate_arg_(self, value):
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
        aa = None
        if isinstance(value, Marker):
            assert self._is_comparable_to_(value)
            aa = value.change_frame(self).coordinates

        # handle dataframes
        elif isinstance(value, pd.DataFrame):
            txt = "'value' index is not comparable to 'self' index."
            assert self.sample_number == value.shape[0], txt
            assert np.sum(self.time - value.index.to_numpy()) == 0, txt
            txt = "'value' columns are not the same as 'self' coordinates."
            assert value.shape[1] == self.dimension_number, txt
            cols = value.columns.to_numpy()
            assert np.all([i in cols for i in self.dim]), txt
            aa = value.values
            try:
                aa = aa.astype(float)
            except Exception:
                ValueError("All elements of 'value' must be numeric.")

        # handle numpy arrays
        elif isinstance(value, (np.ndarray, list)):
            aa = np.atleast_2d(value)
            if aa.shape[1] == 1:
                aa = np.hstack([aa for _ in np.arange(self.dimension_number)])
            elif aa.shape[0] == 1:
                aa = np.vstack([aa for _ in np.arange(self.sample_number)])
            txt = "value's shape is not compatible with '+' operator."
            assert aa.shape[0] == self.sample_number and aa.shape[1] == self.dimension_number, txt
            try:
                aa = aa.astype(float)
            except Exception:
                ValueError("All elements of 'value' must be numeric.")

        # handle scalar values
        elif isinstance(value, (float, int)):
            aa = np.atleast_2d([value])
            aa = np.hstack([aa for _ in np.arange(self.dimension_number)])
            aa = np.vstack([aa for _ in np.arange(self.sample_number)])

        # objects of other instances are not supported
        else:
            TypeError("'value' type not supported for this operation.")

        # return the array
        return aa


    def __add__(self, value):
        vv = self.copy()
        vv.coordinates = vv.coordinates + self._validate_arg_(value)
        return vv


    def __radd__(self, value):
        return self.__add__(value)


    def __iadd__(self, value):
        self.coordinates += self._validate_arg_(value)


    def __sub__(self, value):
        V = self.copy()
        V.coordinates = V.coordinates - self._validate_arg_(value)
        return V


    def __rsub__(self, value):
        return self.__sub__(value)


    def __isub__(self, value):
        self.coordinates -= self._validate_arg_(value)


    def __mul__(self, value):
        V = self.copy()
        V.coordinates *= self._validate_arg_(value)
        return V


    def __rmul__(self, value):
        return self.__mul__(value)


    def __imul__(self, value):
        self.coordinates *= self._validate_arg_(value)


    def __truediv__(self, value):
        V = self.copy()
        V.coordinates /= self._validate_arg_(value)
        return V


    def __rtruediv__(self, value):
        return self.__truediv__(value)


    def __itruediv__(self, value):
        self.coordinates /= self._validate_arg_(value)


    def __neg__(self):
        V = self.copy()
        V.coordinates *= (-1)
        return V


    def __pow__(self, value):
        txt = "power operator is allowed only for float or int exponents."
        assert isinstance(value, (float, int)), txt
        V = self.copy()
        V.coordinates ** value
        return V


    def __abs__(self):
        V = self.copy()
        V.coordinates = abs(V.coordinates)
        return V


    def __eq__(self, other):
        if isinstance(other, Marker):
            return self.to_df() == other.to_df()
        return False


    def __ne__(self, other):
        return not self.__eq__(other)


    def change_frame(self, R):
        """
        view the current object from another Reference Frame.

        Input

            R (ReferenceFrame)

                the ReferenceFrame object on which the current Marker
                has to be aligned.

        Output

            M (Marker)

                the marker represented by the ReferenceFrame R
        """

        # check the reference frame
        txt = "'R' must be a ReferenceFrame like object."
        assert self._is_comparable_to_(R), txt

        # move the coordinates of the actual marker to the "global" frame
        actual_frame = Rotation.from_matrix(self.orientation)
        actual_obj = actual_frame.inv().apply(self.coordinates) + self.origin
        target_obj = Rotation.from_matrix(R.orientation).apply(actual_obj - R.origin)
        return Marker(
                coordinates = target_obj,
                origin = R.origin,
                orientation = R.orientation,
                sampling_frequency = self.sampling_frequency,
                names = self.dimensions,
                unit = R.unit
                )


    def as_Vector(self):
        """
        return a copy of the current Marker as Vector instance.
        """
        return Vector(
                coordinates = self.coordinates,
                origin = self.origin,
                orientation = self.orientation,
                sampling_frequency = self.sampling_frequency,
                names = self.dimensions,
                unit = self.unit
                )


    def as_ReferenceFrame(self):
        """
        return the ReferenceFrame instance of the current object.
        """
        return ReferenceFrame(
                origin = self.origin,
                orientation = self.orientation,
                sampling_frequency = self.sampling_frequency,
                names = self.dimensions,
                unit = self.unit
                )


    def to_df(self):
        """
        generate a pandas.DataFrame representing the Marker.
        """
        cols = ["{} ({})".format(d, self.unit) for d in self.dimensions]
        index = pd.Index(self.time, name = "Time (s)")
        cord = pd.DataFrame(self.coordinates, columns = cols, index = index)
        return pd.concat([cord, super().to_df()], axis = 1)


    def copy(self):
        """
        return a copy of the object.
        """
        return Marker(
                coordinates = self.coordinates,
                origin = self.origin,
                orientation = self.orientation,
                sampling_frequency = self.sampling_frequency,
                names = self.dimensions,
                unit = self.unit
                )


class Vector(Marker):
    """
    Create a Vector object instance.

    Input

        coordinates (numpy 2 dimensional array, pandas.DataFrame)

            the coordinates of the marker within the provided ReferenceFrame.
            each row denotes one single sample.
            if a pandas.DataFrame is provided, the columns are used as
            names of the dimensions.

        origin (numpy 2 dimensional array)

            a numpy array containing the origin of the "local"
            reference frame. 'origin' must have shape N x D where N are the
            number of samples and D the number of dimensions.


        orientation (numpy 3 dimensional array)

            a 3D numpy array with shape N x D x D, where N are the number of
            samples and D the number of dimensions. Please note that the 2nd
            dimension defines the orientation, while the 3rd the coordinates of
            each versor in the provided dimensions.

        sampling_frequency (float)

            the sampling frequency (in Hz) at which the samples have been
            collected.

        names (list, numpy 1 dimensional array)

            the names of each dimension provided as array or list of strings.

        unit (str)

            the unit of measurement of the ReferenceFrame.
    """


    def __init__(
            self,
            coordinates = np.array([[0, 0, 0]]),
            origin = None,
            orientation = None,
            sampling_frequency = 1,
            names = None,
            unit = "m"
            ):
        super().__init__(coordinates, origin, orientation, sampling_frequency, names, unit)


    def as_Marker(self):
        """
        return a copy of the current object as Marker instance.
        """
        return Marker(
                coordinates = self.coordinates,
                origin = self.origin,
                orientation = self.orientation,
                fs = self.sampling_frequency,
                names = self.dimensions,
                unit = self.unit
                )


    def cross(self, value):
        """
        get the cross product between 3D Vectors.

        Input:

            value (Vector)

                a 3D vector to be cross multiplied to self.
        """
        # check value
        return (Vector(
                coordinates = np.cross(self.coordinates, self._validate_arg_(value)),
                sampling_frequency = self.sampling_frequency,
                origin = self.origin.values,
                orientation = self.orientation.values,
                names = self.dimensions,
                unit = self.unit
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
                data = np.sqrt(np.sum(self.coordinates ** 2, 1)),
                columns = ["||" + "+".join(self.dimensions) + "||"],
                index = self.time
                )


    def angle_from(self, B, return_matrix = False, degrees = True):
        """
        obtain the angle allowing to rotate B into self.
        Optionally return also the unit vector around which the rotation occurs.

        Input

            B (Vector)

                the vector from which the angle has to be calculated.
                If B has a different ReferenceFrame than self, it is firstly
                aligned to it.

            return_matrix (bool)

                if True, return the rotation matrix rotating B into self.

            degrees (bool)

                if True the output angle is returned with degrees as unit.
                Otherwise, it is returned in radians

        Output

            A (pandas.DataFrame)

                the angle from B to self.

            R (dict, optional)

                if return_matrix is True, a dict is returned where each key
                will be a time sample of self and B and its argument the
                rotation matrix allowing to rotate B into self
                by post multiplication.
        """

        # check B
        assert isinstance(B, Vector), "'B' must be a Vector object."
        K = B.change_frame(self)

        # get both B and self have length > 0
        an = self.norm()
        bn = K.norm()
        txt = "All samples of '{}' must have norm > 0."
        assert np.all(an.values > 0), txt.format('self')
        assert np.all(bn.values > 0), txt.format('B')

        # check the other options
        txt = "'{}' must be a 'bool' object."
        assert return_matrix or not return_matrix, txt.format("return_matrix")
        assert degrees or not degrees, txt.format("degrees")

        # get the angle
        cn = (K - self).norm()
        A = np.arccos((an ** 2 + bn ** 2 - cn ** 2) / (2 * an * bn))
        if degrees:
            unit = " (deg)"
            A *= 180 / np.pi
        else:
            unit = " (rad)"
        A.columns = pd.Index(['Angle{}'.format(unit)])

        # return the angle
        if not return_matrix:
            return A

        # get the (normalized) rotation vector and
        # handle the case that self = -B
        a = self / an.values
        b = K / bn.values
        w = b.cross(a)
        inv = np.isclose(np.sum((a + b).coordinates, 1), 0)
        inv = np.argwhere(inv).flatten()
        n = w.norm().values
        n[inv] = 1
        w = w / n

        # get the rotations
        R = Rotation.from_rotvec(w.coordinates * A).as_matrix()

        # handle the non-definite cases
        R[inv] = -np.eye(self.dimension_number).astype(int)

        # return the set of rotations
        R = {i: R[j] for j, i in enumerate(self.time)}
        return A, R


    def copy(self):
        """
        return a copy of the object.
        """
        return Vector(
                coordinates = self.coordinates,
                sampling_frequency = self.sampling_frequency,
                origin = self.origin,
                orientation = self.orientation,
                names = self.dimensions,
                unit = self.unit
                )


    def change_frame(self, R):
        """
        view the current object from another Reference Frame.

        Input

            R (ReferenceFrame)

                the ReferenceFrame object on which the current Vector
                has to be aligned.

        Output

            M (Vector)

                the vector represented by the ReferenceFrame R
        """
        return super().change_frame(R).as_Vector()


class ForcePlatform(Vector):
    """
    Create a Force object instance.

    Input

        coordinates (numpy 2 dimensional array, pandas.DataFrame)

            the coordinates of the marker within the provided ReferenceFrame.
            each row denotes one single sample.
            if a pandas.DataFrame is provided, the columns are used as
            names of the dimensions.

        origin (numpy 2 dimensional array)

            a numpy array containing the origin of the "local"
            reference frame. 'origin' must have shape N x D where N are the
            number of samples and D the number of dimensions.


        orientation (numpy 3 dimensional array)

            a 3D numpy array with shape N x D x D, where N are the number of
            samples and D the number of dimensions. Please note that the 2nd
            dimension defines the orientation, while the 3rd the coordinates of
            each versor in the provided dimensions.

        sampling_frequency (float)

            the sampling frequency (in Hz) at which the samples have been
            collected.

        names (list, numpy 1 dimensional array)

            the names of each dimension provided as array or list of strings.

        unit (str)

            the unit of measurement of the ReferenceFrame.
    """


    def __init__(
            self,
            coordinates = np.array([[0, 0, 0]]),
            origin = np.array([[0, 0, 0]]),
            orientation = np.array(
                    [[[1, 0, 0],
                      [0, 1, 0],
                      [0, 0, 1]]]
                    ),
            sampling_frequency = 1,
            names = np.array(['X', 'Y', 'Z']),
            unit = "m"
            ):
        super().__init__(coordinates, origin, orientation, sampling_frequency, names, unit)


    def copy(self):
        """
        return a copy of the object.
        """
        return ForcePlatform(
                coordinates = self.coordinates,
                sampling_frequency = self.sampling_frequency,
                origin = self.origin,
                orientation = self.orientation,
                names = self.dimensions,
                unit = self.unit
                )


    def change_frame(self, R):
        """
        view the current object from another Reference Frame.

        Input

            R (ReferenceFrame)

                the ReferenceFrame object on which the current Vector
                has to be aligned.

        Output

            M (Vector)

                the vector represented by the ReferenceFrame R
        """
        return super().change_frame(R).as_Vector()


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


    def to_excel(self, path, new_file = False):
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
    def from_excel(path, sheets = None, exclude_errors = True):
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
                    data = K,
                    index = time,
                    time_unit = time_unit,
                    dim_unit = dim_unit,
                    type = type
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
