
# IMPORTS

from base import *
import os
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation


# CLASSES


class ReferenceFrame():

    def __init__(self, origin = None, versors = None, unit = "m", fs = 1):
        """
        Create a ReferenceFrame instance.

        Input

            origin (2D numpy.ndarray or pandas.DataFrame)

                a numpy array containing the origin of the "local"
                reference frame.

            versors (dict of pandas.DataFrame of 2D numpy.ndarray(s))

                a list of len equal to the number of keys of origin.
                Each dict must have same keys of origin with one numeric value
                per key.
        """

        # check the origin
        assert isinstance(origin, dict), "'origin' must be a dict."

        # check each versor
        assert len(versors) == len(origin), str(
            len(origin)) + " versors are required."
        dims = np.array([i for i in origin])
        V = []
        for versor in versors:
            assert np.all([j in dims for j in versor]
                          ), "all versors must have keys: " + str(dims)
            vec = _UnitDataFrame(versor, index=[
                0], type="versor", dim_unit="", time_unit="")
            V += [vec / vec.norm.values]

        # store the data
        self.origin = origin
        self.versors = V

        # calculate the rotation matrix
        self._to_global = np.vstack([i.values for i in V])
        self._to_local = np.linalg.inv(self._to_global)

    def _build_origin(self, vector):
        """
        Internal method used to build a Vector with the same index of "vector" containing the
        origin coordinates.

        Input:
            vector: (pyomech.Vector)
                    the vector used to mimimc the shape of the output orgin vector.

        Output:
            O:      (pyomech.Vector)
                    the vector representing the origin at each index of vector.
        """

        # ensure vector is a vector
        assert _UnitDataFrame.match(
            vector), "'vector' must be an instance of pyomech.Vector."

        # build O
        O = {i: np.tile(self.origin[i], vector.shape[0]) for i in self.origin}
        return _UnitDataFrame(O, index=vector.index, time_unit=vector.time_unit, dim_unit=vector.dim_unit,
                             type="Coordinates")

    def to_local(self, vector):
        """
        Rotate "vector" from the current "global" ReferenceFrame to the "local" ReferenceFrame
        defined by the current instance of this class.

        Input:
            vector: (pyomech.Vector)
                    the vector to be aligned.

        Output:
            v_loc:  (pyomech.Vector)
                    the same vector with coordinates aligned to the current "local" ReferenceFrame.
        """

        # ensure vector can be converted
        O = self._build_origin(vector)
        assert _UnitDataFrame.match(
            O, vector), "'vector' cannot be aligned to the local ReferenceFrame."

        # rotate vector
        V = vector - O
        V.loc[V.index] = V.values.dot(self._to_local)
        return V

    def to_global(self, vector):
        """
        Rotate "vector" from the current "local" ReferenceFrame to the "global" ReferenceFrame.

        Input:
            vector: (pyomech.Vector)
                    the vector to be aligned.

        Output:
            v_glo:  (pyomech.Vector)
                    the same vector with coordinates aligned to the "global" ReferenceFrame.
        """

        # ensure vector can be converted
        O = self._build_origin(vector)
        assert _UnitDataFrame.match(
            O, vector), "'vector' cannot be aligned to the local ReferenceFrame."

        # rotate vector
        V = vector.copy()
        V.loc[V.index] = V.values.dot(self._to_global)
        return V + O


class _UnitDataFrame(pd.DataFrame):
    """
    Create n-dimensional point sampled over time.

    Input:

        data (ndarray, list, pandas.Series, pandas.DataFrame, dict)

            the data which creates the Point.

        index (list)

            the index representing each time sample. It must provide 1 value
            for each sample in data.

        columns (list)

            the names of the dimensions of the points.

        dim_unit (str)

            the unit of measurement of each dimension

        time_unit (str)

            the unit of measurement of the samples
    """
    _metadata = ["time_unit", "dim_unit", "type"]

    # CONVERTERS

    def to_dict(self):
        """
        return the data as dict.
        """
        return {d: self[d].values.flatten() for d in self.columns}

    def to_df(self):
        """
        Store the Point into a "pandas DataFrame" formatted having a column
        named "Index_ZZZ" and the others as "XXX|YYY_ZZZ" where:

            'XXX' the type of the point
            'YYY' the dimension of the point
            'ZZZ' the dim_unit
        """

        # create the Point df
        v_df = pd.DataFrame(
            data=self.values,
            columns=[self.type + "|" + i + "_" + self.dim_unit
                     for i in self.columns]
        )

        # add the index column
        v_df.insert(0, 'Index_' + self.time_unit, self.index.to_numpy())

        # return the df
        return v_df

    def to_csv(self, file, **kwargs):
        """
        Store the Point into a "csv". The file is formatted having a column
        named "Index_ZZZ" and the others as "XXX|YYY_ZZZ" where:

            'XXX' the type of the point
            'YYY' the dimension of the point
            'ZZZ' the dim_unit

        Input:

            file (str)

                the file path.
        """

        # ensure the file can be stored
        os.makedirs(lvlup(file), exist_ok=True)

        # store the output data
        try:
            kwargs.pop('index')
        except Exception:
            pass
        try:
            kwargs.pop('path')
        except Exception:
            pass
        self.to_df().to_csv(file, index=False, **kwargs)

    def to_excel(self, file, sheet="Sheet1", new_file=False):
        """
        Store the Point into an excel file sheet. The file is formatted
        having a column named "Index_ZZZ" and the others as "XXX|YYY_ZZZ" where:

            'XXX' the type of the point
            'YYY' the dimension of the point
            'ZZZ' the dim_unit

        Input:

            file (str)

                the file path.

            sheet (str or None)

                the sheet name.

            new_file (bool)

                should a new file be created rather than adding the current
                point to an existing one?
        """
        return to_excel(file, self.to_df(), sheet, new_file)

    # CLASS SPECIFIC METHODS

    def fs(self):
        """
        get the mean sampling frequency of the Point in Hz.
        """
        return 1. / np.mean(np.diff(self.index.to_numpy()))

    def applyc(self, fun, *args, **kwargs):
        """
        apply a given function to all columns of the Point.

        Input:

            fun (function)

                the function to be applied.
                Please note that each column is passed as first argument to fun.

            args/kwargs

                function arguments that are directly passed to fun.

        Output:

            V (Point)

                The point with the function applied to each row.
        """
        V = self.copy()
        for d in self.columns:
            V.loc[V.index, [d]] = fun(
                self[d].values.flatten(), *args, **kwargs)

    def applyr(self, fun, *args, **kwargs):
        """
        apply a given function to all samples of the Point.

        Input:

            fun (function)

                the function to be applied.
                Please note that each row is passed as first argument to fun.

            args/kwargs

                function arguments that are directly passed to fun.

        Output:

            V (Point)

                The point with the function applied to each column.
        """
        V = self.copy()
        for i in V.index:
            V.loc[i, V.columns] = fun(self.loc[i].values, *args, **kwargs)
        return V

    def applya(self, fun, *args, **kwargs):
        """
        apply a given function to all values of the point as one.

        Input:

            fun (function)

                the function to be applied.
                Please note that each row is passed as first argument to fun.

            args/kwargs

                function arguments that are directly passed to fun.

        Output:

            V (Point)

                The point with the function applied to all values.
        """
        V = self.copy()
        V.loc[V.index, V.columns] = fun(self.values, *args, **kwargs)
        return V

    def replace_missing(self, replacement=None, x=None, *args, **kwargs):
        """
        replace missing values.

        Input

            replacement (dict or None, default = None)

                a dict mapping the replacing value of each dimension to a key.

            x (list(_Point) or None)

                if replacement is None, Linear Regression is used to predict
                the values to replace the missing data.
                If this is the case, 'x' represents the list of _Point objects
                to be used as Indipendent variables for the prediction.
                These objects must be provided as list. The index of each
                Point must mimic the index of the actual object.

            *args, **kwargs (any)

                the additional parameters to be passed to LinearRegression.

        Output

            F (UnitDataFrame subclass)

                A new object with the missing values being replaced.
        """
        # ensure missing data exist
        missing_index = self.loc[self.isna().any(1)].index
        if missing_index.shape[0] == 0:
            return self.copy()

        # use the provided replacement (if any)
        F = self.copy()
        if replacement is not None:

            # validate the replacement argument
            _validate_obj(replacement, (dict))
            alert_dim = 'The current object does not have dimension {}'
            alert_len = '{} must be of len = 1'
            keys = np.array([i for i in replacement])
            for key in keys:
                if isinstance(replacement[key], (np.ndarray, list)):
                    assert len(replacement[key]) == 1, alert_len.format(key)
                    replacement[key] = replacement[key][0]
                _validate_obj(replacement[key], (float, int))

            # replace the missing values
            replacement = [replacement[key] for key in self.columns]
            replacement = np.atleast_2d(replacement)
            replacement = np.vstack([replacement for i in missing_index])
            F.loc[missing_index, F.columns] = replacement
            return F

        # get the set of valid values
        valid_index = [i for i in self.index.values.flatten()
                       if i not in missing_index.values.flatten()]
        valid_index = pd.Index(valid_index)

        # validate x
        _validate_obj(x, (list))
        X = []
        for key in x:
            _validate_obj(x[key], _UnitDataFrame)
        X = pd.concat(X, axis=1)

        # obtain the regression coefficients
        LM = LinearRegression(
            y=self.loc[valid_index, self.columns],
            x=X.loc[valid_index, X.columns],
            *args, **kwargs
        )

        # replace the missing values
        pred = LM.predict(X.loc[missing_index, X.columns])
        F.loc[missing_index, F.columns] = pred
        return F

    # PRIVATE METHODS

    def __init__(self, *args, **kwargs):

        # remove special class objects
        meta_props = {}
        for prop in self._metadata:
            try:
                meta_props[prop] = kwargs.pop(prop)
            except Exception:
                pass

        # handle Series props
        ser_props = {}
        for prop in ["name", "fastpath"]:
            try:
                ser_props[prop] = kwargs.pop(prop)
            except Exception:
                pass

        # generate the pandas object
        if len(ser_props) > 0:
            super().__init__(pd.Series(*args, **ser_props))
        else:
            super().__init__(*args, **kwargs)

        # add the extra features
        for prop in self._metadata:
            try:
                pr = meta_props[prop]
            except Exception:
                pr = ""
            setattr(self, prop, pr)
        self.type = self.__class__.__name__

    def __finalize__(self, other, method=None):
        """propagate metadata from other to self """

        # merge operation: using metadata of the left object
        if method == "merge":
            for name in self._metadata:
                object.__setattr__(self, name, getattr(
                    other.left, name, getattr(self, name)))

        # concat operation: using metadata of the first object
        elif method == "concat":
            for name in self._metadata:
                object.__setattr__(self, name, getattr(
                    other.objs[0], name, getattr(self, name)))

        # any other condition
        else:
            for name in self._metadata:
                object.__setattr__(self, name, getattr(
                    other, name, getattr(self, name)))
        return self

    def __str__(self):
        out = pd.DataFrame(self)
        idx = [i + self.time_unit for i in self.index.to_numpy()]
        out.index = pd.Index(idx)
        col = [i + self.dim_unit for i in out.columns.to_numpy()]
        out.columns = pd.Index(col)
        out = out.__str__() + "\ntype: {}".format(self.type)
        return out

    def __repr__(self):
        return self.__str__()

    def __getattr__(self, *args, **kwargs):
        try:
            out = super(_UnitDataFrame, self).__getattr__(*args, **kwargs)
            return out.__finalize__(self)
        except Exception:
            AttributeError()

    def __getitem__(self, *args, **kwargs):
        try:
            out = super(_UnitDataFrame, self).__getitem__(*args, **kwargs)
            return out.__finalize__(self)
        except Exception:
            NotImplementedError()

    @property
    def _constructor(self):
        return _UnitDataFrame

    @property
    def _constructor_sliced(self):
        return _UnitDataFrame

    @property
    def _constructor_expanddim(self):
        return _UnitDataFrame


class Point(_UnitDataFrame):
    """
    Create Point object sampled over time.

    Input:

        data (ndarray, list, pandas.Series, pandas.DataFrame, dict)

            the data which creates the Point.

        index (list)

            the index representing each time sample. It must provide 1 value
            for each sample in data.

        columns (list)

            the names of the dimensions of the points.

        dim_unit (str)

            the unit of measurement of each dimension

        time_unit (str)

            the unit of measurement of the samples
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.type = "Point"

    @property
    def _constructor(self):
        return Point

    @property
    def _constructor_sliced(self):
        return Point

    @property
    def _constructor_expanddim(self):
        return Point

    # valid operators
    def __add__(self, value):
        valid = (Point, Vector, float, int, np.float, np.int,
                 np.ndarray)
        _validate_obj(value, valid)
        if isinstance(value, (np.ndarray)):
            r, c = value.shape
            ck_1 = (r == 1) & (c == self)
        return super().__add__(value).__finalize__(self)



class Vector(_UnitDataFrame):
    """
    Create a Vector object sampled over time.

    Input:

        data (ndarray, list, pandas.Series, pandas.DataFrame, dict)

            the data which creates the Point.

        index (list)

            the index representing each time sample. It must provide 1 value
            for each sample in data.

        columns (list)

            the names of the dimensions of the points.

        dim_unit (str)

            the unit of measurement of each dimension

        time_unit (str)

            the unit of measurement of the samples

        type (str)

            a string describing the nature of the Point object.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.type = "Vector"

    @property
    def _constructor(self):
        return Vector

    @property
    def _constructor_sliced(self):
        return Vector

    @property
    def _constructor_expanddim(self):
        return Vector

    # CLASS SPECIFIC METHODS

    def norm(self):
        """
        Get the norm of the point.
        """
        return _UnitDataFrame(
            data=(self ** 2).sum(1).values.flatten() ** 0.5,
            index=self.index,
            columns=["|" + " + ".join(self.columns) + "|"],
            time_unit=self.time_unit,
            dim_unit=self.dim_unit,
            type=self.type
        )


class ForcePlatform(Vector):
    """
    Create a Vector object sampled over time.

    Input:

        data (ndarray, list, pandas.Series, pandas.DataFrame, dict)

            the data which creates the Point.

        index (list)

            the index representing each time sample. It must provide 1 value
            for each sample in data.

        columns (list)

            the names of the dimensions of the points.

        dim_unit (str)

            the unit of measurement of each dimension

        time_unit (str)

            the unit of measurement of the samples

        type (str)

            a string describing the nature of the Point object.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.type = "ForcePlatform"

    @property
    def _constructor(self):
        return ForcePlatform

    @property
    def _constructor_sliced(self):
        return ForcePlatform

    @property
    def _constructor_expanddim(self):
        return ForcePlatform


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


def gram_schmidt(normalized=False, *args, **kwargs):
    """
    return the orthogonal basis defined by a set of points using the
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

    # check the input
    _validate_obj(normalized, bool)
    D = {**kwargs}
    keys = np.array([i for i in kwargs.keys()])
    n_args = np.arange(len(args))
    names = ["V" + str(i + 1) + ("_1" if i in keys else "") for i in n_args]
    D.update(**{i: j for i, j in zip(names, args)})
    txt = "All input data must be Points with equal index and columns."
    assert _UnitDataFrame.match(**D), txt

    # internal function to simplify projection calculation
    def proj(a, b):
        aa = a.values
        bb = b.values
        return np.inner(aa, bb) / np.inner(bb, bb) * bb

    # calculate the projection points
    keys = np.array([i for i in D])
    W = {keys[0]: D[keys[0]]}
    for i in np.arange(1, len(D)):
        W[keys[i]] = D[keys[i]]
        for j in np.arange(i):
            W[keys[i]] -= proj(D[keys[i]], D[keys[j]])

    # normalize if required
    if normalized:
        for key in W:
            W[key] /= W[key].norm.values

    # return the output
    return W


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
