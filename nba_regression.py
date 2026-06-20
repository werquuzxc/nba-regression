import numpy as np
seasons  = np.array([1,    2,    3,    4,    5   ])
points   = np.array([19.3, 21.3, 24.6, 25.9, 27.0])
usage    = np.array([24.1, 26.3, 29.8, 31.2, 33.1])
X = np.column_stack([
    seasons - seasons.mean(),
    usage   - usage.mean(),
])
w  = np.array([0.0, 0.0])
b  = 0.0
lr = 0.01
epochs = 1000
for i in range(epochs):
    pred  = X @ w + b
    error = pred - points
    mse = np.mean(error ** 2)
    dw = X.T @ error / len(error)
    db = np.mean(error)
    w -= lr * dw
    b -= lr * db
print(w, b, mse)
next_season   = 6
next_usage    = 34.5
next_X = np.array([
    next_season - seasons.mean(),
    next_usage  - usage.mean(),
])
prediction = next_X @ w + b
print(prediction)
