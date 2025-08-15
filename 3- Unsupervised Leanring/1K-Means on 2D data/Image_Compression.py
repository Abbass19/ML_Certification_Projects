import numpy as np
import matplotlib.pyplot as plt
from utils import *
from functions import (find_closest_centroids,compute_centroids,run_kMeans,
                       kMeans_init_centroids,Image_Compression)

original_img = plt.imread('bird_small.png')

Image_Compression(original_img,16,10)



# #We need to change X_img
# for i in range(len(centroids)):
#     Mask = (idx==i)
#     X_img[Mask] = centroids[i]
#
# X_out=np.reshape(X_img,128,3)
# print(f'The shape of the image :{X_out.shape} ')
# plt.imshow(X_out)
# plt.show()