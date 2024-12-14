import numpy as np
import matplotlib.pyplot as plt
from utils import *
from functions import compute_cost,compute_gradient,gradient_descent
import copy
import math



# load the dataset
x_train, y_train = load_data()


initial_w = 2
initial_b = 1
cost = compute_cost(x_train, y_train, 2, 1)
print(type(cost))
print(f'Cost at initial w: {cost:.3f}')

initial_w = 0
initial_b = 0

tmp_dj_dw, tmp_dj_db = compute_gradient(x_train, y_train, initial_w, initial_b)
print('Gradient at initial w, b (zeros):', tmp_dj_dw, tmp_dj_db)

# Compute and display cost and gradient with non-zero w
test_w = 0.2
test_b = 0.2
tmp_dj_dw, tmp_dj_db = compute_gradient(x_train, y_train, test_w, test_b)

print('Gradient at test w, b:', tmp_dj_dw, tmp_dj_db)


initial_w = 0.
initial_b = 0.

# some gradient descent settings
iterations = 1500
alpha = 0.01

w,b,_,_ = gradient_descent(x_train ,y_train, initial_w, initial_b,
                     compute_cost, compute_gradient, alpha, iterations)
print("w,b found by gradient descent:", w, b)

m = x_train.shape[0]
predicted = np.zeros(m)


for i in range(m):
    predicted[i] = w * x_train[i] + b

