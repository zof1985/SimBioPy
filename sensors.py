# SENSORS MODULE


#! IMPORTS

from . import geometry as gm
import numpy as np
import pandas as pd
import plotly.express as px


#! CLASSES


class Marker3D(gm.Point):
    """
    generate a 3D object reflecting a marker collected over time in a 3D space.

    Parameters
    ----------
    coordinates: Point, pandas.DataFrame, numpy.ndarray, list
        the amplitude of the vector.

    index: arraylike
        the index for both the amplitude and origin of the vector.
    """

    def __init__(self, coordinates, index=None):
        amp = self._get_data(coordinates, index, ["X", "Y", "Z"])
        assert amp.ndim == 3, "a 3D dataset must be provided."
        super(Marker3D, self).__init__(coordinates=amp)


class ForcePlatform3D(gm._Object):
    """
    generate a 3D object reflecting a force platform whose data have been
    collected over time in a 3D space.

    Parameters
    ----------
    force: Vector, _DF, Point, pandas.DataFrame, numpy.ndarray, list
        the amplitude of the force vector.

    moment: Vector, _DF, Point, pandas.DataFrame, numpy.ndarray, list
        the amplitude of the moments vector.

    origin: _DF, Point, pandas.DataFrame, numpy.ndarray, list
        the origin of the force and moments vectors.

    index: arraylike | None
        the index for the input data.
    """

    @property
    def force_vectors(self):
        """
        return the force vector measured by the ForcePlaftorm.
        """
        return gm.Vector(amplitude=self._force, origin=self._origin)

    @property
    def moment_vectors(self):
        """
        return the moments vector measured by the ForcePlaftorm.
        """
        return gm.Vector(amplitude=self._moment, origin=self._origin)

    def __init__(self, force, moment, origin, index=None):
        # handle the case force and moment are vectors
        if isinstance(force, gm.Vector) or isinstance(moment, gm.Vector):
            txt = "force and moment vectors must match and have the same origin."
            assert isinstance(force, gm.Vector) and isinstance(moment, gm.Vector), txt
            assert force.matches(moment), txt
            assert np.sum((force.origin - moment.origin).values) == 0, txt
            frz = force.amplitude
            mnt = moment.amplitude
            ori = force.origin
        else:
            frz = force.coordinates if isinstance(force, gm.Point) else force
            mnt = moment.coordinates if isinstance(moment, gm.Point) else moment
            ori = origin.coordinates if isinstance(origin, gm.Point) else origin

        # check the size
        txt = "only 3D data can be provided."
        assert frz.ndim == 3, txt
        assert mnt.ndim == 3, txt
        assert ori.ndim == 3, txt
        frz = self._get_data(frz, index, ["X", "Y", "Z"])
        mnt = self._get_data(mnt, index, ["X", "Y", "Z"])
        ori = self._get_data(ori, index, ["X", "Y", "Z"])

        # build the object
        super(ForcePlatform3D, self).__init__(
            force=frz,
            moment=mnt,
            origin=ori,
            index=index,
        )


class Link3D(gm.Segment):
    """
    Generate an object reflecting a dimension-less vector in a n-dimensional space.

    Parameters
    ----------

    p0, p1: Point, _DF, pandas.DataFrame, numpy.ndarray, list
        the first and second points of the segment.

    index: arraylike
        the index for both the amplitude and origin of the vector.

    """

    def __init__(self, p0, p1, index=None):
        v0 = p0.coordinates if isinstance(p0, gm.Point) else p0
        assert v0.ndim == 3, "a 3D dataset must be provided."
        v0 = self._get_data(v0, index, ["X", "Y", "Z"])
        v1 = p1.coordinates if isinstance(p1, gm.Point) else p1
        assert v1.ndim == 3, "a 3D dataset must be provided."
        v1 = self._get_data(v1, index, ["X", "Y", "Z"])
        super(Link3D, self).__init__(p0=v0, p1=v1)


class EmgSensor(gm._MathObject):
    """
    Generate a n-channels EMG sensor instance.

    Parameters
    ----------
    amplitude: _DF, pandas.DataFrame, numpy.ndarray, list
        the amplitude of the EMG signal(s).

    index: arraylike
        the index for both the amplitude and origin of the vector.
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
        elif isinstance(amp, (pd.DataFrame, gm._Df)):
            df = self._get_data(amp)
        super(EmgSensor, self).__init__(amplitude=amp)

    def _math_value(self, obj, transpose: bool = False) -> np.ndarray:
        """
        Parameters
        ----------
        obj: int, float, np.ndarray, _DF, pd.DataFrame, Vector
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


class Imu3D(gm._Object):
    """
    generate a 3D object reflecting an Inertial Measurement Unit
    whose data have been collected over time in a 3D space.

    Parameters
    ----------
    accelerometer: _DF, Point, pandas.DataFrame, numpy.ndarray, list, None
        the amplitude of the accelerations.

    gyroscope: _DF, Point, pandas.DataFrame, numpy.ndarray, list, None
        the amplitude of angular velocities.

    magnetometer: _DF, Point, pandas.DataFrame, numpy.ndarray, list, None
        the amplitude of the magnetic field readings.

    index: arraylike | None
        the index for the input data.
    """

    def __init__(self, accelerometer, gyroscope, magnetometer, index=None):
        attrs = {}
        if accelerometer is not None:
            if isinstance(accelerometer, gm.Point):
                attrs["accelerometer"] = accelerometer.coordinates
            else:
                attrs["accelerometer"] = accelerometer
        if gyroscope is not None:
            if isinstance(gyroscope, gm.Point):
                attrs["gyroscope"] = gyroscope.coordinates
            else:
                attrs["gyroscope"] = gyroscope
        if magnetometer is not None:
            if isinstance(magnetometer, gm.Point):
                attrs["magnetometer"] = magnetometer.coordinates
            else:
                attrs["magnetometer"] = magnetometer

        # check the size
        txt = "only 3D data can be provided."
        for i, v in attrs.items():
            assert v.ndim == 3, txt
            attrs[i] = self._get_data(v, index, ["X", "Y", "Z"])

        # build the object
        super(Imu3D, self).__init__(index=index, **attrs)
