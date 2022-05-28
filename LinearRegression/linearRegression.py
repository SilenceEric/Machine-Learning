from operator import mod
from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt

X = [[10.0], [8.0], [13.0], [9.0], [11.0], [14.0], [6.0], [4.0], [12.0], [7.0], [5.0]]
y = [8.04, 6.95, 7.58, 8.81, 8.33, 9.96, 7.24, 4.26, 10.84, 4.82, 5.68]

model = LinearRegression()
model.fit(X, y)
#斜率
k = model.coef_ 
#截距
b = model.intercept_
x1 = np.linspace(0,15,100)
y1 = k*x1 + b

plt.scatter(X, y)
plt.plot(x1, y1)
plt.show()