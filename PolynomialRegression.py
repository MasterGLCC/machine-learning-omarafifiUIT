import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures     
X = np.array([-3, -2, -1, 0, 1, 2, 3]).reshape(-1, 1)
y = np.array([9, 4, 1, 0, 1, 4, 9])                        
poly_converter = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly_converter.fit_transform(X)
model = LinearRegression()
model.fit(X_poly, y)
X_smooth = np.linspace(-3, 3, 100).reshape(-1, 1)
X_smooth_poly = poly_converter.transform(X_smooth)
y_smooth_pred = model.predict(X_smooth_poly)

plt.scatter(X, y, color='red', s=50, label='data')
plt.plot(X_smooth, y_smooth_pred, color='green', linewidth=3, label='curve')
plt.title('Polynomialreg')
plt.legend()
plt.show()