import numpy as np
import matplotlib.pyplot as plt
from utils import *
import os



def find_closest_centroids(X, centroids):
    K = centroids.shape[0]
    idx = np.zeros(X.shape[0], dtype=int)
    for i in range(len(X)):
        distance = np.zeros(K)
        for j in range(K):
            array = (X[i]-centroids[j])
            distance[j] = np.dot(array,array)
        idx[i] = np.argmin(distance)
    return idx



def compute_centroids(X, idx, K):
    rows, columns = X.shape
    centroids = np.zeros((K, columns))
    for i in range(K):
        Mask = (idx==i)
        centroid = np.mean(X[Mask],axis=0)
        centroids[i,:] = centroid
    return centroids


# You do not need to implement anything for this part

def run_kMeans(X, initial_centroids, max_iters=10, plot_progress=True):

    # Initialize values
    row, columns = X.shape
    K = initial_centroids.shape[0]
    centroids = initial_centroids
    previous_centroids = centroids
    idx = np.zeros(row)
    plt.figure(figsize=(8, 6))

    # Run K-Means
    for i in range(max_iters):
        print("K-Means iteration %d/%d" % (i, max_iters - 1))
        idx = find_closest_centroids(X, centroids)
        if plot_progress:
            plot_progress_kMeans(X, centroids, previous_centroids, idx, K, i)
            previous_centroids = centroids
        centroids = compute_centroids(X, idx, K)
    # plt.show()
    return centroids, idx


def kMeans_init_centroids(X, K):
    randidx = np.random.permutation(X.shape[0])
    centroids = X[randidx[:K]]
    return centroids


def Image_Compression(original_img, K, max_iterations):
    X_img = np.reshape(original_img, (original_img.shape[0] * original_img.shape[1], 3))
    Initial_file_size_bytes = X_img.nbytes
    Initial_file_size__kb_bytes=Initial_file_size_bytes/1024
    initial_centroids = kMeans_init_centroids(X_img, K)
    centroids, idx = run_kMeans(X_img, initial_centroids, max_iterations)
    idx = find_closest_centroids(X_img, centroids)
    X_recovered = centroids[idx, :]
    X_recovered = np.reshape(X_recovered, original_img.shape)
    Final_file_size_bytes =X_recovered.nbytes
    Final_file_size_kb_bytes = Final_file_size_bytes/1024


    ratio_reduction = Initial_file_size__kb_bytes/Final_file_size_kb_bytes
    fig, ax = plt.subplots(1, 2, figsize=(16, 16))
    plt.axis('off')
    ax[0].imshow(original_img)
    ax[0].set_title('Original')
    ax[0].set_axis_off()
    ax[1].imshow(X_recovered)
    ax[1].set_title(f'Compressed with {K} colours ratio reduction {ratio_reduction}' )
    ax[1].set_axis_off()
    plt.show()
    return ratio_reduction

# def reduction_graph_maker(initial_K, final_K):
#     reduction_ratio = np.zeros(final_K-initial_K)
#     original_img = plt.imread('bird_small.png')
#     X= np.arange(initial_K,final_K)
#
#     for i in range(initial_K,final_K):
#         reduction_ratio[i-initial_K]=Image_Compression(original_img,i,10)
#     plt.plot(X,reduction_ratio)
#     plt.title(f"The reduction ratio for different K selection")
#     plt.show()
#
#
# def max_iteration_graph_maker(initial, final):
#     reduction_ratio = np.zeros(final - initial)
#     original_img = plt.imread('bird_small.png')
#     max_iterations = np.arange(initial, final)
#     for i in range(initial,final):
#         reduction_ratio[i-initial]=Image_Compression(original_img,16,i)
#     plt.plot(X, reduction_ratio)
#     plt.title(f"The reduction ratio for different itera   tions for selection")
#     plt.show()
#
#
#


