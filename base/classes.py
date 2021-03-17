
# IMPORTS

from typing import overload
import numpy as np
import pandas as pd
import os
from . import functions as fn
from scipy.spatial.transform import Rotation


# CLASSES

class Vector(pd.DataFrame):
    """
    class representing an n-dimensional vector sampled over time.
    """



    # CLASS PROPERTIES

    time_unit = ""
    dim_unit = ""
    type = ""
    _metadata = ["time_unit", "dim_unit", "type"]



    @property
    def module(self):
        """
        Get the module of the vector.
        """
        return Vector(
            data      = (self ** 2).sum(1).values.flatten() ** 0.5,
            index     = self.index,
            columns   = ["|" + " + ".join(self.columns) + "|"],
            time_unit = self.time_unit,
            dim_unit  = self.dim_unit,
            type      = self.type
            )



    @property
    def fs(self):
        """
        get the mean sampling frequency of the Vector in Hz.
        """

        return 1. / np.mean(np.diff(self.index.to_numpy()))



    # STATIC METHODS

    @staticmethod
    def angle_by_3_vectors(A, B, C):
        """
        return the angle ABC using the Cosine theorem.

        Input:
            A:  (Vector)
                The coordinates of one vector.

            B:  (Vector)
                The coordinates of the vector over which the angle has
                to be calculated.

            C:  (Vector)
                The coordinates of the third vector.

        Output:
            K:  (Vector)
                A 1D vector containing the result of:
                                     2          2            2
                           /  (A - B)  +  (C - B)  -  (A - C)  \
                    arcos | ----------------------------------- |
                           \      2 * (A - B) * (C - B)        /
        """

        # ensure all entered parameters are vectors
        txt = "'A', 'B' and 'C' must be Vectors with equal index and columns."
        assert Vector.match(A, B, C), txt

        # get a, b and c
        a = (A - B).module
        b = (C - B).module
        c = (A - C).module

        # return the angle
        k = (a ** 2 + b ** 2 - c ** 2) / (2 * a * b).values
        k.loc[k.index] = np.arccos(k.values)
        k.time_unit = A.time_unit
        k.dim_unit = "rad"
        k.type = "Angle"
        return k



    @staticmethod
    def gram_schmidt(normalized=False, *args, **kwargs):
        """
        return the orthogonal basis defined by a set of vectors using the
        Gram-Schmidt algorithm.

        Input:

            normalized (bool)

                should the projected vectors returned in normalized units?

            args / kwargs (Vector)

                one or more vectors from which the orthogonal projections have
                to be calculated.

        Output:

            W (VectorDict)

                a VectorDict object containing the orthogonal vectors.
        """

        # check the input
        fn._validate_obj(normalized, bool)
        D = {**kwargs}
        keys = np.array([i for i in kwargs.keys()])
        D.update(**{"V{}".format(i) + ("" if i not in keys else "_1"): args[i]
                    for i in np.arange(len(args))})
        txt = "All input data must be Vectors with equal index and columns."
        assert Vector.match(**D), txt

        # internal function to simplify projection calculation
        def proj(a, b):
            aa = a.values
            bb = b.values
            return np.inner(aa, bb) / np.inner(bb, bb) * bb

        # calculate the projection vectors
        keys = np.array([i for i in D])
        W = {keys[0]: D[keys[0]]}
        for i in np.arange(1, len(D)):
            W[keys[i]] = D[keys[i]]
            for j in np.arange(i):
                W[keys[i]] -= proj(D[keys[i]], D[keys[j]])

        # normalize if required
        if normalized:
            for key in W:
                W[key] /= W[key].module.values

        # return the output
        return W



    @staticmethod
    def match(*args, **kwargs):
        """
        check if the entered objects are instance of Vector or
        pandas.DataFrame. If more than one parameter is provided, check
        also that all the entered objects have the same columns and indices.

        Output:
            C (bool)

                True if all inputs are Vectors or a pandas.DataFrame with the
                same columns and index of self. False, otherwise.
        """

        # get the elements entered
        objs = [i for i in args] + [kwargs[i] for i in kwargs]

        # check if all elements are instance of Vector or DataFrame
        for obj in objs:
            if not isinstance(obj, (Vector, pd.DataFrame)):
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
            if not np.all([col_check, idx_check, shp_check]):
                return False
        return True



    @staticmethod
    def read_csv(*args, **kwargs):
        """
        return the Vector from a "csv". The file is formatted having a column
        named "Index_ZZZ" and the others as:

        "XXX|YYY_ZZZ" where:
            'XXX' the type of the vector
            'YYY' the dimension of the vector
            'ZZZ' the dim_unit

        Input:

            arguments to be passed to the pandas "read_csv" function:
            https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_csv.html

        Output:

            V (Vector)

                the imported vector.
        """
        return Vector.from_csv(*args, **kwargs)



    @staticmethod
    def from_df(df):
        """
        return the Vector from a pandas DataFrame. The df is formatted having
        a column named "Index_ZZZ" and the others as "XXX|YYY_ZZZ" where:

            'XXX' the type of the vector
            'YYY' the dimension of the vector
            'ZZZ' the dim_unit

        Input:

            df (pandas.DataFrame)

                the input pandas.DataFrame object

        Output:

            v (Vector)

                the imported vector.
        """

        # get the index
        idx_col = [i for i in df.columns if "_".join(i.split("_")[:-1])][0]
        idx_val = df[idx_col].values.flatten()

        # get the time_unit
        time_unit = idx_col.split("_")[-1]

        # remove the index column
        df = df[[i for i in df.columns if i != idx_col]]

        # get the vector type
        typ = np.unique(["|".join(i.split("|")[:-1]) for i in df.columns])
        txt = "No vector type has been found" if len(typ) == 0 else str(
            len(typ)) + " vector types have been found."
        assert len(typ) == 1, txt
        typ = typ[0]

        # get the dim_unit
        uni = np.unique([i.split("_")[-1] for i in df.columns])
        txt = "No unit type has been found" if len(uni) == 0 else str(
            len(uni)) + " dimension units have been found."
        assert len(uni) == 1, txt
        uni = uni[0]

        # update the columns
        df.columns = ["_".join(i.split("|")[-1].split("_")[:-1])
                      for i in df.columns]

        # get the vector
        return Vector(
            data      = df.to_dict("list"),
            index     = idx_val,
            time_unit = time_unit,
            dim_unit  = uni,
            type      = typ
            )



    @staticmethod
    def from_csv(*args, **kwargs):
        """
        return the Vector from a "csv". The file is formatted having a column
        named "Index_ZZZ" and the others as:

        "XXX|YYY_ZZZ" where:
            'XXX' the type of the vector
            'YYY' the dimension of the vector
            'ZZZ' the dim_unit

        Input:

            arguments to be passed to the pandas "read_csv" function:
            https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_csv.html

        Output:

            V (Vector)

                the imported vector.
        """
        return Vector.from_df(pd.read_csv(*args, **kwargs))



    @staticmethod
    def from_excel(file, sheet, *args, **kwargs):
        """
        return the Vector from an excel file. The file is formatted having
        a column named "Index_ZZZ" and the others as "XXX|YYY_ZZZ" where:

            'XXX' the type of the vector
            'YYY' the dimension of the vector
            'ZZZ' the dim_unit

        Input:

            file (str)

                the path to the file

            sheet (str)

                the sheet to be imported

            args, kwargs:

                additional parameters passed to pandas.read_excel

        Output:

            v (Vector)

                the imported vector.
        """
        return Vector.from_df(
            fn.from_excel(file, sheet, *args, **kwargs)[sheet]
            )



    def to_dict(self):
        """
        return the data as dict.
        """
        return {d: self[d].values.flatten() for d in self.columns}



    def to_df(self):
        """
        Store the Vector into a "pandas DataFrame" formatted having a column
        named "Index_ZZZ" and the others as "XXX|YYY_ZZZ" where:

            'XXX' the type of the vector
            'YYY' the dimension of the vector
            'ZZZ' the dim_unit
        """

        # create the Vector df
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
        Store the Vector into a "csv". The file is formatted having a column
        named "Index_ZZZ" and the others as "XXX|YYY_ZZZ" where:

            'XXX' the type of the vector
            'YYY' the dimension of the vector
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
        Store the Vector into an excel file sheet. The file is formatted
        having a column named "Index_ZZZ" and the others as "XXX|YYY_ZZZ" where:

            'XXX' the type of the vector
            'YYY' the dimension of the vector
            'ZZZ' the dim_unit

        Input:

            file (str)

                the file path.

            sheet (str or None)

                the sheet name.

            new_file (bool)

                should a new file be created rather than adding the current
                vector to an existing one?
        """

        fn.to_excel(file, self.to_df(), sheet, new_file)



    def apply_col(self, fun, *args, **kwargs):
        """
        apply a given function to all columns of the Vector.

        Input:

            fun (function)

                the function to be applied.
                Please note that each column is passed as first argument to fun.

            args/kwargs

                function arguments that are directly passed to fun.

        Output:

            V (Vector)

                The vector with the function applied to each row.
        """

        return Vector(
            data      = {d: fun(self[d].values.flatten(), *args, **kwargs)
                         for d in self.columns},
            index     = self.index,
            dim_unit  = self.dim_unit,
            time_unit = self.time_unit,
            type      = self.type
            )



    def apply_row(self, fun, *args, **kwargs):
        """
        apply a given function to all samples of the Vector.

        Input:

            fun (function)

                the function to be applied.
                Please note that each row is passed as first argument to fun.

            args/kwargs

                function arguments that are directly passed to fun.

        Output:

            V (Vector)

                The vector with the function applied to each column.
        """
        V = self.copy()
        for i in V.index:
            V.loc[i, V.columns] = fun(self.loc[i].values, *args, **kwargs)
        return V



    def apply_mat(self, fun, *args, **kwargs):
        """
        apply a given function to all values of the vector as one.

        Input:

            fun (function)

                the function to be applied.
                Please note that each row is passed as first argument to fun.

            args/kwargs

                function arguments that are directly passed to fun.

        Output:

            V (Vector)

                The vector with the function applied to all values.
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
            super(Vector, self).__init__(pd.Series(*args, **ser_props))
        else:
            super(Vector, self).__init__(*args, **kwargs)

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
        return Vector



    @property
    def _constructor_sliced(self):
        return Vector



    @property
    def _constructor_expanddim(self):
        return Vector



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
            out = super(Vector, self).__getattr__(*args, **kwargs)
            return out.__finalize__(self)
        except Exception:
            AttributeError()



class VectorDict(dict):
    """
    Create a dict of "Vector" object(s). It is a simple wrapper of the "dict" class object with additional methods.

    Input:
        args: (objects)
            objects of class vectors.
    """

    def to_csv(self, path, **kwargs):
        """
        store pandas.DataFrames containing the vectors formatted as: "XXX|YYY_ZZZ".

        where:
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

    @staticmethod
    def from_csv(path, **kwargs):
        """
        Create a "VectorDict" object from a "csv" or "txt" file.

        Input:
            path: (str)
                an existing ".csv" or "txt" file or a folder containing csv files. The files must contain 1 column
                named "Index_ZZZ" and the others as "WWW:XXX|YYY_ZZZ" where:
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
        vd = VectorDict()

        # check if the path is a file or a folder and populate the VectorDict accordingly
        if os.path.isfile(path):
            vd[".".join(path.split(os.path.sep)[-1].split(".")[:-1])
               ] = Vector.from_csv(path, **kwargs)
        else:
            for i in get_files(path, ".csv", False):
                vd[".".join(i.split(os.path.sep)[-1].split(".")[:-1])
                   ] = Vector.from_csv(i, **kwargs)

        # return the dict
        return vd

    @staticmethod
    def from_excel(path, sheets=None, exclude_errors=True):
        """
        Create a "VectorDict" object from an excel file.

        Input:
            path:           (str)
                            an existing excel file. The sheets must contain 1 column named "Index_ZZZ" and the
                            others as "WWW:XXX|YYY_ZZZ" where:
                                'WWW' is the type of the vector
                                'XXX' is the name of the vector
                                'YYY' is the dimension of the vector
                                'ZZZ'  if the dim_unit.

            sheets:         (str, list or None)
                            the sheets to be imported. In None, all sheets are imported.

            exclude_errors: (bool)
                            If a sheet generates an error during the import would you like to skip it and import the
                            others?

        Output:
            a new VectorDict with the imported vectors.
        """

        vd = VectorDict()

        # get the sheets
        dfs = from_excel(path, sheets)

        # import the sheets
        for i in dfs:
            if exclude_errors:
                try:
                    vd[i] = Vector.from_df(dfs[i])
                except Exception:
                    pass
            else:
                vd[i] = Vector.from_df(dfs[i])

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
        vd = VectorDict()

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
            K = {i if i != "" else v: values[rows,
                                             np.argwhere(V == v + (("." + i)
                                                                   if i != "" else "")).flatten()].flatten() for i in D}

            # setup the output variable
            vd[v] = Vector(K, index=time, time_unit=time_unit,
                           dim_unit=dim_unit, type=type)

        # return vd
        return vd

    # SUBCLASSED METHODS

    def __init__(self, *args, **kwargs):
        super(VectorDict, self).__init__(*args, **kwargs)
        self.__finalize__

    # this method is casted at the end of each manipulating action in order to check that only Vector objects
    # are stored into the dict

    @property
    def __finalize__(self):
        for i in self.keys():
            assert self[i].__class__.__name__ == "Vector", i + \
                " is not a Vector object"
        return self

    def __str__(self):
        return "\n".join([" ".join(["\n\nVector:\t ", i, "\n\n", self[i].__str__()]) for i in self.keys()])

    def __repr__(self):
        return self.__str__()

    def __setitem__(self, *args, **kwargs):
        super(VectorDict, self).__setitem__(*args, **kwargs)
        self.__finalize__

    def __setattr__(self, *args, **kwargs):
        super(VectorDict, self).__setattr__(*args, **kwargs)
        self.__finalize__


# Reference Frame
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
            V += [vec / vec.module.values]

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
