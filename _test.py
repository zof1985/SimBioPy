# REGRESSION MODULE
from simbiopy.regression import *

# LINEAR REGRESSION
x = np.arange(10)
y = x * 1.5 + 0.5 + np.random.randn(len(x)) * 0.01
lr = LinearRegression(y=y, x=x)
print(lr)
print(lr(x))
lrf = LinearRegression(y=y, x=x, fit_intercept=False)
print(lrf)
print(lrf(x))

# POLYNOMIAL REGRESSION
pr = PolynomialRegression(y=y, x=x, n=2)
print(pr)
print(pr(x))
prf = PolynomialRegression(y=y, x=x, n=2, fit_intercept=False)
print(prf)
print(prf(x))
