import numpy as np
import tensorflow as tf

def cofi_cost_func(X, W, b, Y, R, lambda_):
    nm, nu = Y.shape
    J_value=0
    b = b.flatten()
    for j in range(nu):
        for i in range(nm):
            if R[i,j]:
                value_function = np.dot(W[j, :], X[i, :])
                value_function += b[j]
                value_function -= Y[i, j]
                value_function **= 2
                value_function *= 0.5
                J_value +=value_function

    # Regularization terms
    weight_regularization = (lambda_ / 2) * np.sum(W ** 2)
    feature_regularization = (lambda_ / 2) * np.sum(X ** 2)

    return J_value + weight_regularization + feature_regularization



##This function makes the same as the ones before it however it is
#made with linear algebra to be quicker. Now made my day.
#Although I thought before that such thing is not doable
def cofi_cost_func_v(X, W, b, Y, R, lambda_):
    j = (tf.linalg.matmul(X, tf.transpose(W)) + b - Y)*R
    J = 0.5 * tf.reduce_sum(j**2) + (lambda_/2) * (tf.reduce_sum(X**2) + tf.reduce_sum(W**2))
    return J

