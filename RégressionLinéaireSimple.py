import numpy as np
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
X = np.array([[1], [2], [3], [4], [5]]) 
y = np.array([2, 4, 5, 4, 5])
modele_simple = LinearRegression()
modele_simple.fit(X, y)
predictions = modele_simple.predict(X)
print(f"Équation : y = {modele_simple.coef_[0]:.2f}x + {modele_simple.intercept_:.2f}")

plt.scatter(X, y, color='red', s=50, label='x')
plt.plot(X, y, color='green', linewidth=3, label='y')
plt.title('simplyyyyyyyyyyyyyyy')
plt.legend()
plt.show()