import numpy as np
import matplotlib.pyplot as plt

X = np.array([
    [2, 3],
    [3, 4],
    [5, 6],
    [6, 7],
    [1, 1],
    [2, 1]
])

y = np.array([0, 0, 1, 1, 0, 0])

X = np.hstack((np.ones((X.shape[0], 1)), X))


def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def cost(X, y, w):
    m = len(y)
    h = sigmoid(X @ w)
    return -(1/m) * np.sum(y*np.log(h) + (1-y)*np.log(1-h))

def gradient_descent(X, y, w, lr, iters):
    m = len(y)
    costs = []

    for _ in range(iters):
        h = sigmoid(X @ w)
        gradient = (1/m) * X.T @ (h - y)
        w = w - lr * gradient
        costs.append(cost(X, y, w))

    return w, costs


w = np.zeros(X.shape[1])
w_final, costs = gradient_descent(X, y, w, lr=0.1, iters=200)


plt.figure(figsize=(12,5))


plt.subplot(1,2,1)
plt.plot(costs)
plt.title("Coût")
plt.xlabel("Itérations")
plt.ylabel("Log Loss")
plt.grid()

plt.subplot(1,2,2)


X0 = X[y == 0]
X1 = X[y == 1]

plt.scatter(X0[:,1], X0[:,2], color='red', label='Classe 0')
plt.scatter(X1[:,1], X1[:,2], color='blue', label='Classe 1')

x1 = np.array([X[:,1].min(), X[:,1].max()])
x2 = -(w_final[0] + w_final[1]*x1) / w_final[2]

plt.plot(x1, x2, 'k--', label='Frontière')

plt.title("Régression Logistique")
plt.xlabel("X1")
plt.ylabel("X2")
plt.legend()
plt.grid()

plt.show()



pred = sigmoid(X @ w_final) >= 0.5
accuracy = np.mean(pred == y)

print("Accuracy :", accuracy)