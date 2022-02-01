# SENSORS MODULE


#! IMPORTS


from .geometry import *
import numpy as np
import pandas as pd
import plotly.express as px
import warnings


#! CLASSES


class Sensor:
    """
    general class containing just methods defining that the GeometricObject is
    also a Sensor.
    """

    def __init__(self):
        pass


class Marker3D(Point, Sensor):
    """
    generate a 3D object reflecting a marker collected over time in a 3D space.

    Parameters
    ----------
    coordinates: Point, pandas.DataFrame, numpy.ndarray, list
        the amplitude of the vector.

    index: arraylike
        the index for the samples of the sensor.

    unit: str
        the unit of measurement of the sensors's coordinates.
    """

    def __init__(self, coordinates, index=None, unit="m"):
        amp = self._get_data(
            coordinates=coordinates,
            index=index,
            columns=["X", "Y", "Z"],
            unit=unit,
        )
        assert amp.ndim == 3, "a 3D dataset must be provided."
        super(Marker3D, self).__init__(coordinates=amp, unit=unit)


class ForcePlatform3D(GeometricObject, Sensor):
    """
    generate a 3D object reflecting a force platform whose data have been
    collected over time in a 3D space.

    Parameters
    ----------
    force: Vector, UnitDataFrame, Point, pandas.DataFrame, numpy.ndarray, list
        the amplitude of the force vector.

    moment: Vector, UnitDataFrame, Point, pandas.DataFrame, numpy.ndarray, list
        the amplitude of the moments vector.

    origin: UnitDataFrame, Point, pandas.DataFrame, numpy.ndarray, list
        the origin of the force and moments vectors.

    index: arraylike | None
        the index for the input data.

    force_unit: str
        the unit of measurement of the force data.

    moment_unit: str
        the unit of measurement of the moment data.

    origin_unit: str
        the unit of measurement of the origin data.
    """

    @property
    def force_vectors(self):
        """
        return the force vector measured by the ForcePlaftorm.
        """
        return Vector(amplitude=self._force, origin=self._origin)

    @property
    def moment_vectors(self):
        """
        return the moments vector measured by the ForcePlaftorm.
        """
        return Vector(amplitude=self._moment, origin=self._origin)

    def __init__(
        self,
        force,
        moment,
        origin,
        index=None,
        force_unit="N",
        moment_unit="Nm",
        origin_unit="m",
    ):
        # handle the case force and moment are vectors
        if isinstance(force, Vector) or isinstance(moment, Vector):
            txt = "force and moment vectors must match and have the same "
            txt += "origin."
            assert isinstance(force, Vector), txt
            assert isinstance(moment, Vector), txt
            assert force.matches(moment, strict=False), txt
            assert np.sum((force.origin - moment.origin).values) == 0, txt
            frz = force.amplitude
            mnt = moment.amplitude
            ori = force.origin
        else:
            frz = force.coordinates if isinstance(force, Point) else force
            mnt = moment.coordinates if isinstance(moment, Point) else moment
            ori = origin.coordinates if isinstance(origin, Point) else origin

        # check the size
        txt = "only 3D data can be provided."
        assert frz.ndim == 3, txt
        assert mnt.ndim == 3, txt
        assert ori.ndim == 3, txt
        frz = self._get_data(
            data=frz,
            index=index,
            columns=["X", "Y", "Z"],
            unit=force_unit,
        )
        mnt = self._get_data(
            data=mnt,
            index=index,
            columns=["X", "Y", "Z"],
            unit=moment_unit,
        )
        ori = self._get_data(
            data=ori,
            index=index,
            columns=["X", "Y", "Z"],
            unit=origin_unit,
        )

        # build the object
        super(ForcePlatform3D, self).__init__(
            force=frz,
            moment=mnt,
            origin=ori,
        )


class Link3D(Segment, Sensor):
    """
    Generate an object reflecting a dimension-less vector in a n-dimensional
    space.

    Parameters
    ----------

    p0, p1: Point, UnitDataFrame, pandas.DataFrame, numpy.ndarray, list
        the first and second points of the segment.

    index: arraylike
        the index for the samples of the sensor.

    unit: str
        the unit of measurement of the sensors.
    """

    def __init__(self, p0, p1, index=None, unit=""):
        v0 = p0.coordinates if isinstance(p0, Point) else p0
        assert v0.ndim == 3, "a 3D dataset must be provided."
        v0 = self._get_data(v0, index, ["X", "Y", "Z"], unit)
        v1 = p1.coordinates if isinstance(p1, Point) else p1
        assert v1.ndim == 3, "a 3D dataset must be provided."
        v1 = self._get_data(v1, index, ["X", "Y", "Z"], unit)
        super(Link3D, self).__init__(p0=v0, p1=v1)


class EmgSensor(GeometricMathObject, Sensor):
    """
    Generate a n-channels EMG sensor instance.

    Parameters
    ----------
    amplitude: UnitDataFrame, pandas.DataFrame, numpy.ndarray, list
        the amplitude of the EMG signal(s).

    index: arraylike
        the index for the samples of the sensor.

    unit: str
        the unit of measurement of the sensors.
    """

    def __init__(self, amplitude, index=None):
        """
        constructor
        """
        if isinstance(amplitude, list):
            amp = np.array(list)
        else:
            amp = amplitude
        if isinstance(amp, np.ndarray):
            if amp.ndim == 1:
                amp = np.atleast_2d(amp).T
            elif amp.ndim > 2:
                raise ValueError("amplitude must be a 1D or 2D dataset.")
            cols = ["Channel{}".format(i + 1) for i in range(amp.shape[1])]
            df = self._get_data(amp, index, cols)
        elif isinstance(amp, (pd.DataFrame, UnitDataFrame)):
            df = self._get_data(amp)
        super(EmgSensor, self).__init__(amplitude=df)

    def _math_value(self, obj, transpose: bool = False) -> np.ndarray:
        """
        Parameters
        ----------
        obj: int, float, np.ndarray, UnitDataFrame, pd.DataFrame, Vector
            the second object included in the math operation.

        transpose: bool (optional, default=False)
            if True, the transpose of obj is checked. Otherwise the
            obj data is controlled as is.

        Returns
        -------
        val: np.ndarray
            the value to be used for the math operation.
        """
        if isinstance(obj, EmgSensor):
            val = obj.amplitude
        else:
            val = obj
        return super(EmgSensor, self)._math_value(val, transpose)

    def __add__(self, obj):
        """
        addition.
        """
        return EmgSensor(amplitude=self.amplitude + self._math_value(obj))

    def __sub__(self, obj):
        """
        subtraction.
        """
        return EmgSensor(amplitude=self.amplitude - self._math_value(obj))

    def __mul__(self, obj):
        """
        multiplication
        """
        return EmgSensor(amplitude=self.amplitude * self._math_value(obj))

    def __truediv__(self, obj):
        """
        division
        """
        return EmgSensor(amplitude=self.amplitude / self._math_value(obj))

    def __floordiv__(self, obj):
        """
        floor division
        """
        return EmgSensor(amplitude=self.amplitude // self._math_value(obj))

    def __pow__(self, obj):
        """
        power elevation (** operator).
        """
        return EmgSensor(amplitude=self.amplitude ** self._math_value(obj))

    def __mod__(self, obj):
        """
        module (% operator).
        """
        return EmgSensor(amplitude=self.amplitude % self._math_value(obj))

    def __matmul__(self, obj):
        """
        matrix multiplication (@ operator).
        """
        val = self._math_value(obj, transpose=True)
        return EmgSensor(amplitude=self.amplitude @ val)


class Imu3D(GeometricMathObject, Sensor):
    """
    generate a 3D object reflecting an Inertial Measurement Unit
    whose data have been collected over time in a 3D space.

    Parameters
    ----------
    accelerometer: Any of the valid data input (see below)
        the amplitude of the accelerations.

    gyroscope: Any of the valid data input (see below)
        the amplitude of angular velocities.

    magnetometer: Any of the valid data input (see below)
        the amplitude of the magnetic field readings.

    index: arraylike
        the index for the samples of the sensor.

    accelerometer_unit: str
        the unit of measurement of the accelerometer.

    gyroscope_unit: str
        the unit of measurement of the gyroscope.

    magnetometer_unit: str
        the unit of measurement of the magnetometer.

    Valid inputs
    ------------
    UnitDataFrame, Point, pandas.DataFrame, numpy.ndarray, list, None
    """

    def __init__(
        self,
        accelerometer,
        gyroscope,
        magnetometer,
        index=None,
        accelerometer_unit="m/s^2",
        gyroscope_unit="rad/s",
        magnetometer_unit="nT",
    ):
        attrs = {}
        txt = "only 3D data can be provided."
        if accelerometer is not None:
            if isinstance(accelerometer, Point):
                attrs["accelerometer"] = accelerometer.coordinates
                assert attrs["accelerometer"].ndim == 3, txt
            else:
                attrs["accelerometer"] = self._get_data(
                    data=accelerometer,
                    index=index,
                    coordinates=["X", "Y", "Z"],
                    unit=accelerometer_unit,
                )
        if gyroscope is not None:
            if isinstance(gyroscope, Point):
                attrs["gyroscope"] = gyroscope.coordinates
                assert attrs["gyroscope"].ndim == 3, txt
            else:
                attrs["gyroscope"] = self._get_data(
                    data=gyroscope,
                    index=index,
                    coordinates=["X", "Y", "Z"],
                    unit=gyroscope_unit,
                )
        if magnetometer is not None:
            if isinstance(magnetometer, Point):
                attrs["magnetometer"] = magnetometer.coordinates
                assert attrs["magnetometer"].ndim == 3, txt
            else:
                attrs["magnetometer"] = self._get_data(
                    data=magnetometer,
                    index=index,
                    coordinates=["X", "Y", "Z"],
                    unit=magnetometer_unit,
                )

        # build the object
        super(Imu3D, self).__init__(**attrs)


class Model3D:
    """
    generic class used as interface for the implementation of
    specific 3D models.

    Parameters
    ----------
    sensors: keyworded arguments
        The list of arguments containing the data of the object.
        The key of the arguments will be used as attributes of the object.
        The values of each key must be an instance of the Sensor class.
    """

    # list of attributes of the class which contain relevant data.
    # NOTE THESE ATTRIBUTES ARE CONSTANTS THAT SHOULD NOT BE MODIFIED
    _attributes = []

    def __init__(self, **sensors):
        """
        class constructor.
        """
        self._attributes = []

        # populate the object with the input sensors
        for attr, value in sensors.items():
            self.append(obj=value, name=attr)

    def __str__(self) -> str:
        """
        convert self to a string.
        """
        return self.pivot().__str__()

    def append(self, obj, name):
        """
        append an object to self and store it appropriately.

        Parameters
        ----------
        obj: GeometricObject instance
            the object to be stored.

        name: str
            the label denoting the object.
        """

        # check the input data
        txt = "obj must be an instance of the GeometricObject class."
        assert isinstance(obj, Sensor), txt
        assert isinstance(name, str), "name must be a string."
        txt = "only 3D objects must be provided."
        assert isinstance(obj, EmgSensor) or obj.ndim == 3, txt

        # ensure the object will be stored according to its class
        cls = obj.__class__.__name__
        if not cls in self._attributes:
            self._attributes += [cls]
            setattr(self, cls, {})

        # check if another object with the same name exists
        if any([i == name for i in getattr(self, cls)]):
            txt = "{} already existing in {}. The old instance has been "
            txt += "replaced"
            warnings.warn(txt.format(name, cls))

        # store the sensor
        getattr(self, cls).update({name: obj})

    def pivot(self) -> pd.DataFrame:
        """
        generate a wide dataframe object containing both origin and
        amplitudes.
        """
        col = ["Sensor", "Type", "Source", "Dimension"]
        df = self.stack().pivot("Time", col)
        df.columns = pd.Index([i[1:] for i in df.columns])
        return df

    def plot(
        self,
        as_subplots: bool = False,
        lines: bool = True,
        show: bool = True,
        width: int = 1280,
        height: int = 720,
    ):
        """
        generate a plotly plot representing the current object.

        Parameters
        ----------
        as_subplots: bool (default=False)
            should the dimensions of object be plotted as a single subplot?

        lines: bool (default=True)
            if True, only lines linking the samples are rendered. Otherwise,
            a scatter plot is rendered.

        show: bool (default=True)
            if True the generated figure is immediately plotted.
            Otherwise the generated object is returned.

        width: int (default=1280)
            the width of the output figure in pixels

        height: int (default=720)
            the height of the output figure in pixels

        Returns
        -------
        None, if show = True. A plotly.Figure object, otherwise.
        """
        fun = px.line if lines else px.scatter
        fig = fun(
            data_frame=self.stack(),
            x="Time",
            y="Amplitude",
            color="Dimension",
            facet_row="Dimension" if as_subplots else None,
            facet_col="Source",
            width=width,
            height=height,
            template="simple_white",
        )
        fig.update_layout(showlegend=not as_subplots)
        if show:
            fig.show()
        else:
            return fig

    def describe(self, percentiles: list = []) -> pd.DataFrame:
        """
        provide descriptive statistics about the parameters in df.

        Parameters
        ----------
        percentiles: list
            a list of values in the [0, 1] range defining the desired
            percentiles to be calculated.

        Returns
        -------
        df: pd.DataFrame
            a pandas.DataFrame with the object descriptive statistics.
        """
        grp = self.stack().drop("Time", axis=1)
        grp = grp.groupby(["Type", "Sensor", "Source", "Dimension"])
        df = grp.describe(percentiles=percentiles)
        df.columns = pd.Index([i[1] for i in df.columns])
        return df

    def stack(self) -> pd.DataFrame:
        """
        stack the object as a long format DataFrame.
        """
        out = []
        for attr in self._attributes:
            for lbl, obj in getattr(self, attr).items():
                df = obj.stack()
                df.insert(0, "Sensor", np.tile(lbl, df.shape[0]))
                df.insert(0, "Type", np.tile(attr, df.shape[0]))
                out += [df]
        return pd.concat(out, axis=0, ignore_index=True)

    def copy(self):
        """
        make a copy of the object
        """
        return self.unstack(self.stack())

    @classmethod
    def unstack(cls, df: pd.DataFrame):
        """
        convert a long format DataFrame into an instance of the object.

        Parameters
        ----------
        df: pandas.DataFrame
            a pandas.DataFrame sorted as it would be generated by the
            .stack() method.

        Returns
        -------
        obj: GeometricObject
            the instance resulting from the dataframe reading.
        """
        out = cls()
        root = ["Type", "Sensor"]
        types = np.unique(df[root].values.astype(str), axis=0)
        for typ, sns in types:
            sub = df.loc[df[root].isin([sns, typ]).all(1)]
            out.append(obj=eval(typ).unstack(sub), name=sns)
        return out

    @property
    def attributes(self):
        """
        return the attributes of this instance.
        """
        return self._attributes
