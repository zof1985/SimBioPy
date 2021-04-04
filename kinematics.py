
# IMPORTS

from base import *
import os
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation
from . import functions as fn



# CLASSES

class Marker(Point):
    """
    Create 3D point in space collected over time.

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



    # CLASS PROPERTIES

    time_unit = "s"
    dim_unit = "m"
    type = "Point 3D"



    # STATIC METHODS

    @staticmethod
    def match(*args, **kwargs):
        """
        check if the entered objects are Marker(s).
        If more than one parameter is provided, check
        also that all the entered objects have the same columns and indices.

        Output:
            C (bool)

                True if all inputs are Points or a pandas.DataFrame with the
                same columns and index of self. False, otherwise.
        """
        # get the elements entered
        objs = [i for i in args] + [kwargs[i] for i in kwargs]

        # check if all elements are instance of Point or DataFrame
        for obj in objs:
            if not isinstance(obj, (Marker)):
                return False

        # check the columns and index of all objs
        IX = objs[0].index.to_numpy()
        CL = objs[0].columns.to_numpy()
        SH = objs[0].shape
        for obj in objs:
            OI = obj.index.to_numpy()
            OC = obj.columns.to_numpy()
            col_check = np.all([i in OC for i in CL])
            idx_check = np.all([i in OI for i in IX])
            shp_check = np.all([i == j for i, j in zip(obj.shape, SH)])
            dim_check = SH[1] == 3
            if not np.all([col_check, idx_check, shp_check, dim_check]):
                return False
        return True



    @staticmethod
    def from_df(df):
        """
        return the Marker from a pandas DataFrame. The df is formatted having
        a column named "Index_ZZZ" and the others as "XXX|YYY_ZZZ" where:

            'XXX' the type of the point
            'YYY' the dimension of the point
            'ZZZ' the dim_unit

        Input:

            df (pandas.DataFrame)

                the input pandas.DataFrame object

        Output:

            v (Point)

                the imported point.
        """
        return Point.from_df(df).__finalize__()



    @staticmethod
    def from_csv(*args, **kwargs):
        """
        return the Point from a "csv". The file is formatted having a column
        named "Index_ZZZ" and the others as:

        "XXX|YYY_ZZZ" where:
            'XXX' the type of the point
            'YYY' the dimension of the point
            'ZZZ' the dim_unit

        Input:

            arguments to be passed to the pandas "read_csv" function:
            https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_csv.html

        Output:

            V (Point)

                the imported point.
        """
        return Point.from_df(pd.read_csv(*args, **kwargs))



    @staticmethod
    def from_excel(file, sheet, *args, **kwargs):
        """
        return the Point from an excel file. The file is formatted having
        a column named "Index_ZZZ" and the others as "XXX|YYY_ZZZ" where:

            'XXX' the type of the point
            'YYY' the dimension of the point
            'ZZZ' the dim_unit

        Input:

            file (str)

                the path to the file

            sheet (str)

                the sheet to be imported

            args, kwargs:

                additional parameters passed to pandas.read_excel

        Output:

            v (Point)

                the imported point.
        """
        return Point.from_df(
            fn.from_excel(file, sheet, *args, **kwargs)[sheet]
            )



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
            data    = self.values,
            columns = [self.type + "|" + i + "_" + self.dim_unit
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
        os.makedirs(fn.lvlup(file), exist_ok=True)

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

        fn.to_excel(file, self.to_df(), sheet, new_file)



    # CLASS SPECIFIC METHODS

    def norm(self):
        """
        Get the norm of the point.
        """
        return Point(
            data      = (self ** 2).sum(1).values.flatten() ** 0.5,
            index     = self.index,
            columns   = ["|" + " + ".join(self.columns) + "|"],
            time_unit = self.time_unit,
            dim_unit  = self.dim_unit,
            type      = self.type
            )



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

        return Point(
            data      = {d: fun(self[d].values.flatten(), *args, **kwargs)
                         for d in self.columns},
            index     = self.index,
            dim_unit  = self.dim_unit,
            time_unit = self.time_unit,
            type      = self.type
            )



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



    # PRIVATE METHODS

    def __init__(self, *args, **kwargs):

        # remove special class objects
        props = {}
        for prop in self._metadata:
            try:
                props[prop] = kwargs.pop(prop)
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
            super(Point, self).__init__(pd.Series(*args, **ser_props))
        else:
            super(Point, self).__init__(*args, **kwargs)

        # add the extra features
        for prop in props:
            setattr(self, prop, props[prop])



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



    @property
    def _constructor(self):
        return Point



    @property
    def _constructor_sliced(self):
        return Point



    @property
    def _constructor_expanddim(self):
        return Point



    def __str__(self):
        out = pd.DataFrame(self).__str__()
        out += "\n".join(["\nAttributes:", "\ttype:\t\t" + self.type,
                          "\ttime_unit:\t" + self.time_unit,
                          "\tdim_unit:\t" + self.dim_unit])
        return out



    def __repr__(self):
        return self.__str__()



    def __getattr__(self, *args, **kwargs):
        try:
            out = super(Point, self).__getattr__(*args, **kwargs)
            return out.__finalize__(self)
        except Exception:
            AttributeError()



class ReferenceFrame():

    def __init__(self, origin, versors):
        """
        Create a ReferenceFrame instance.

        Input:
            origin:     (dict)
                        the coordinates of the origin of a "local" ReferenceFrame. Each key must have
                        a numeric value.

            versors:    (list)
                        a list of len equal to the number of keys of origin. Each dict must have same
                        keys of origin with one numeric value per key.
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
            vec = Vector(versor, index=[
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
        assert Vector.match(
            vector), "'vector' must be an instance of pyomech.Vector."

        # build O
        O = {i: np.tile(self.origin[i], vector.shape[0]) for i in self.origin}
        return Vector(O, index=vector.index, time_unit=vector.time_unit, dim_unit=vector.dim_unit,
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
        assert Vector.match(
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
        assert Vector.match(
            O, vector), "'vector' cannot be aligned to the local ReferenceFrame."

        # rotate vector
        V = vector.copy()
        V.loc[V.index] = V.values.dot(self._to_global)
        return V + O
