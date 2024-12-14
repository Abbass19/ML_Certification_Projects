import numpy as np
import matplotlib.pyplot as plt
from utils import *
from functions import find_closest_centroids,compute_centroids,run_kMeans,kMeans_init_centroids




X = load_data()
print('The shape of X is:', X.shape)


initial_centroids = np.array([[3,3], [6,2], [8,5]])
idx = find_closest_centroids(X, initial_centroids)

# Print closest centroids for the first three elements
print("First three elements in idx are:", idx)

K = 3
centroids = compute_centroids(X, idx, K)

print("The centroids are:", centroids)

run_kMeans(X, initial_centroids, max_iters=100, plot_progress=True)


# Run this cell repeatedly to see different outcomes.

# Set number of centroids and max number of iterations
K = 3
max_iters = 10

# Set initial centroids by picking random examples from the dataset
initial_centroids = kMeans_init_centroids(X, K)

# Run K-Means
centroids, idx = run_kMeans(X, initial_centroids, max_iters, plot_progress=True)