import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

X_multi = np.array([
    [1000, 2], 
    [1500, 3], 
    [2000, 3], 
    [2500, 4]
])
y_multi = np.array([150, 200, 250, 300]) 
modele_multiple = LinearRegression()
modele_multiple.fit(X_multi, y_multi)
nouvelle_maison = np.array([[1800, 3]])
prix_predit = modele_multiple.predict(nouvelle_maison)
print("Coefficients pour chaque variable :", modele_multiple.coef_)
print(f"Prix prédit pour la nouvelle maison : {prix_predit[0]:.2f}")
X_surface = X_multi[:, 0] 
y_pred = modele_multiple.predict(X_multi)
plt.scatter(X_surface, y_multi, color='red', s=50, label='Data')
plt.plot(X_surface, y_pred, color='green', linewidth=3, label='curve')
plt.title('Mulityyyyyyyyyyy (Surface vs Price)')
plt.xlabel('Surface Area')
plt.ylabel('Price')
plt.legend()
plt.show()