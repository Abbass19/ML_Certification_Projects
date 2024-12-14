import numpy
import numpy as np
import math
def sigmoid(z):
    g = np.exp(-z)
    g += 1
    g = 1 / g
    return g


def compute_cost(X, y, w, b, *argv): #Cost function for logistic regression
    m, n = X.shape
    f = np.zeros(X.shape[0])
    loss = np.zeros(X.shape[0])

    for i in range(X.shape[0]):
        tmp = (np.dot(X[i, :], w) + b)
        f[i] = sigmoid(tmp)
        loss[i] = (-y[i] * np.log(f[i]) - (1 - y[i]) * np.log(1 - f[i]))
    total_cost = sum(loss) / (X.shape[0])


    print(f"The value of total loss {total_cost}")
    return total_cost


def compute_gradient(X, y, w, b, *argv):
    m, n = X.shape
    dj_dw = np.zeros(w.shape)
    dj_db = 0.

    for i in range(m):
        f_wb_i = sigmoid(np.dot(X[i], w) + b)
        err_i = f_wb_i - y[i]
        for j in range(n):
            dj_dw[j] = dj_dw[j] + err_i * X[i, j]
        dj_db = dj_db + err_i
    dj_dw = dj_dw / m
    dj_db = dj_db / m
    return dj_db, dj_dw


def gradient_descent(X, y, w_in, b_in, cost_function, gradient_function, alpha, num_iters, lambda_):

    # number of training examples
    m = len(X)

    # An array to store cost J and w's at each iteration primarily for graphing later
    J_history = []
    w_history = []

    for i in range(num_iters):

        # Calculate the gradient and update the parameters
        dj_db, dj_dw = gradient_function(X, y, w_in, b_in, lambda_)

        # Update Parameters using w, b, alpha and gradient
        w_in = w_in - alpha * dj_dw
        b_in = b_in - alpha * dj_db

        # Save cost J at each iteration
        if i < 100000:  # prevent resource exhaustion
            cost = cost_function(X, y, w_in, b_in, lambda_)
            J_history.append(cost)

        # Print cost every at intervals 10 times or as many iterations if < 10
        if i % math.ceil(num_iters / 10) == 0 or i == (num_iters - 1):
            w_history.append(w_in)
            print(f"Iteration {i:4}: Cost {float(J_history[-1]):8.2f}   ")

    return w_in, b_in, J_history, w_history  # return w and J,w history for graphing