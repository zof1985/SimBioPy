# REGRESSION MODULE
from regression import *

# LINEAR REGRESSION
x = np.arange(10)
y = x * 1.5 + 0.5 + np.random.randn(len(x)) * 0.01
lr = LinearRegression(y=y, x=x)
print("\nLINEAR REGRESSION")
print(lr)
print(lr(x))
lrf = LinearRegression(y=y, x=x, fit_intercept=False)
print(lrf)
print(lrf(x))

# POLYNOMIAL REGRESSION
pr = PolynomialRegression(y=y, x=x, n=2)
print("\nPOLYNOMIAL REGRESSION")
print(pr)
print(pr(x))
prf = PolynomialRegression(y=y, x=x, n=2, fit_intercept=False)
print(prf)
print(prf(x))

# POWER REGRESSION
x += 1
y = 0.5 * x**-0.04 + np.random.randn(len(x)) * 0.01
wr = PowerRegression(y=y, x=x)
print("\nPOWER REGRESSION")
print(wr)
print(wr(x))

# HYPERBOLIC REGRESSION
hr = HyperbolicRegression(y=y, x=x)
print("\nHYPERBOLIC REGRESSION")
print(hr)
print(hr(x))

# ELLIPTICAL REGRESSION
er = EllipsisRegression(y=y, x=x)
print("\nELLIPSIS REGRESSION")
print(er)
print(er.axis_major)
print(er.axis_minor)
print(er(x=x))
print(er(y=y))
print(er.eccentricity)
print(er.centre)
