from utils import *
from functions import sigmoid,compute_cost,compute_gradient,gradient_descent

X_train, y_train = load_data("data/ex2data1.txt")


value = 0
print (f"sigmoid({value}) = {sigmoid(value)}")

np.random.seed(1)
initial_w = 0.01 * (np.random.rand(2) - 0.5)
initial_b = -8

# Some gradient descent settings
iterations = 10000
alpha = 0.001

w,b, J_history,_ = gradient_descent(X_train ,y_train, initial_w, initial_b,
                                   compute_cost, compute_gradient, alpha, iterations, 0)