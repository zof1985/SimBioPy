
#! IMPORTS


from numpy import random, linspace
from btsbioengineering import read_tdf
from objects import TimeSeries


#! MAIN


if __name__ == "__main__":
    data = random.randn(100, 3)
    axes = ["X", "Y", "Z"]
    time = linspace(0, 10, data.shape[0])
    unit = "m"
    name = "OBJ"
    obj = TimeSeries(data=data, axes=axes, time=time, unit=unit, name=name)
    obj.iloc[:2] = 1
    obj *= 2
    obj / 4
    tdf = read_tdf("test_sample.tdf")
