import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Loading data
df = pd.read_csv('multiple_linear_regression_dataset.csv')
df = df.values

X_train = df[:, :2]   # 2 features
y_train = df[:, 2]    # target

# Normalizing features
def z_score_normalize(X):
    mu = np.mean(X, axis=0)
    sigma = np.std(X, axis=0)
    X_norm = (X - mu) / sigma
    return X_norm, mu, sigma

X_normal, X_mu, X_std = z_score_normalize(X_train)

# Predicting values
def prediction(X, w, b):
    return np.dot(X, w) + b

# MSE Cost function
def compute_cost(X, y, w, b):
    m = len(y)
    err = prediction(X, w, b) - y
    cost = (1 / (2 * m)) * np.dot(err, err)
    return cost

# Computing Gradient
def compute_gradient(X, y, w, b):
    m, n = X.shape
    dj_dw = np.zeros(n)
    dj_db = 0.

    for i in range(m):
        err = (np.dot(X[i], w) + b) - y[i]
        dj_dw += err * X[i]
        dj_db += err

    dj_dw /= m
    dj_db /= m
    return dj_dw, dj_db

# Gradient Descent Algorithm
def gradient_descent(X, y, w_init, b_init, alpha, num_iter):
    w = copy.deepcopy(w_init)
    b = b_init
    J_history = []

    for i in range(num_iter):
        dj_dw, dj_db = compute_gradient(X, y, w, b)
        w -= alpha * dj_dw
        b -= alpha * dj_db

        cost = compute_cost(X, y, w, b)
        J_history.append(cost)

        if i % (num_iter // 10) == 0:
            print(f"Iteration {i:5d}: Cost {cost:.4e}")

    return w, b, J_history

# Training
w_init = np.zeros(X_train.shape[1])
b_init = 0
alpha = 0.01
num_iter = 1001

found_w, found_b, J_history = gradient_descent(X_normal, y_train, w_init, b_init, alpha, num_iter)

# Plotting Cost
plt.plot(J_history)
plt.title("Cost vs. Iterations")
plt.xlabel("Iteration")
plt.ylabel("Cost (MSE)")
plt.show()

# Regression Plane
from mpl_toolkits.mplot3d import Axes3D

x1 = X_normal[:, 0]
x2 = X_normal[:, 1]
y_pred = prediction(X_normal, found_w, found_b)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x1, x2, y_train, color='blue', label='Actual')
ax.plot_trisurf(x1, x2, y_pred, color='red', alpha=0.5)
ax.set_xlabel('Feature 1 (normalized)')
ax.set_ylabel('Feature 2 (normalized)')
ax.set_zlabel('Target')
ax.set_title('Regression Plane')
plt.legend()
plt.show()