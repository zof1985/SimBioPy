
#! IMPORTS


from numpy import random, linspace
from pandas import concat
from btsbioengineering import read_tdf
from objects import BipolarEMG, TimeSeries


#! MAIN


if __name__ == "__main__":
    data = random.randn(100, 3)
    axes = ["X", "Y", "Z"]
    time = linspace(0, 10, data.shape[0])
    unit = "m"
    name = "OBJ"
    obj = TimeSeries(data=data, dimensions=axes, time=time, unit=unit, name=name, attribute="ATTR",)
    obj2 = obj.copy()
    obj2.iloc[:2] = 1
    obj3 = BipolarEMG(obj2)
    tdf = read_tdf("test_sample.tdf")
