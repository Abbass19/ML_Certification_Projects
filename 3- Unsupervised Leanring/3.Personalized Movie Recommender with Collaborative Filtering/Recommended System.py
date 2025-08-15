import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppresses info and warning messages

import numpy as np
import tensorflow as tf
from tensorflow import keras
from recsys_utils import *
import matplotlib.pyplot as plt
from functions import cofi_cost_func,cofi_cost_func_v

X, W, b, num_movies, num_features, num_users = load_precalc_params_small()
Y, R = load_ratings_small()


movieList, movieList_df = load_Movie_List_pd()
my_ratings_value = np.zeros(num_movies)          #  Initialize my ratings


# Check the file small_movie_list.csv for id of each movie in our dataset
my_ratings_value[2700] = 5
my_ratings_value[2609] = 2
my_ratings_value[929]  = 5
my_ratings_value[246]  = 5
my_ratings_value[2716] = 3
my_ratings_value[1150] = 5
my_ratings_value[382]  = 2
my_ratings_value[366]  = 5
my_ratings_value[622]  = 5
my_ratings_value[988]  = 3
my_ratings_value[2925] = 1
my_ratings_value[2937] = 1
my_ratings_value[793]  = 5


#my_rated is a list of numbers [246, 366, 382, 622, 793, 929, 988, 1150, 2609, 2700, 2716, 2925, 2937]
my_rated = [i for i in range(len(my_ratings_value)) if my_ratings_value[i] > 0]
#Illustrating the Movies been Selected and Rated
print('\nNew user ratings:\n')
for i in range(len(my_ratings_value)):
    if my_ratings_value[i] > 0 :
        print(f'Rated {my_ratings_value[i]} for  {movieList_df.loc[i, "title"]}');





# Reload ratings
Y, R = load_ratings_small()



# Add new user ratings to Y
Y = np.c_[my_ratings_value, Y]

# Add new user indicator matrix to R
R = np.c_[(my_ratings_value != 0).astype(int), R]

# Normalize the Dataset
# Explained in the Session that normalized data makes conversion easier
# It also solves partially the problem of the new user
Ynorm, Ymean = normalizeRatings(Y, R)

#  Useful Values
num_movies, num_users = Y.shape
num_features = 100

# Set Initial Parameters (W, X), use tf.Variable to track these variables
tf.random.set_seed(1234) # for consistent results
W = tf.Variable(tf.random.normal((num_users,  num_features),dtype=tf.float64),  name='W')
X = tf.Variable(tf.random.normal((num_movies, num_features),dtype=tf.float64),  name='X')
b = tf.Variable(tf.random.normal((1,          num_users),   dtype=tf.float64),  name='b')

# Instantiate an optimizer.
optimizer = keras.optimizers.Adam(learning_rate=1e-1)

iterations = 1000
lambda_ = 5
for iter in range(iterations):
    # Use TensorFlow’s GradientTape to record the operations used to compute the cost
    with tf.GradientTape() as tape:
        # Compute the cost (forward pass included in cost)
        cost_value = cofi_cost_func_v(X, W, b, Ynorm, R, lambda_)

    # Use the gradient tape to automatically retrieve
    # the gradients of the trainable variables with respect to the loss
    grads = tape.gradient( cost_value, [X,W,b] )

    # Run one step of gradient descent by updating
    # the value of the variables to minimize the loss.
    optimizer.apply_gradients( zip(grads, [X,W,b]) )

    # Log periodically.
    if iter % 20 == 0:
        print(f"Training loss at iteration {iter}: {cost_value:0.1f}")

# Make a prediction using trained weights and biases
p = np.matmul(X.numpy(), np.transpose(W.numpy())) + b.numpy()

#restore the mean
pm = p + Ymean

my_predictions = pm[:,0]

# sort predictions
ix = tf.argsort(my_predictions, direction='DESCENDING')

for i in range(17):
    j = ix[i]
    if j not in my_rated:
        print(f'Predicting rating {my_predictions[j]:0.2f} for movie {movieList[j]}')

print('\n\nOriginal vs Predicted ratings:\n')
for i in range(len(my_ratings_value)):
    if my_ratings_value[i] > 0:
        print(f'Original {my_ratings_value[i]}, Predicted {my_predictions[i]:0.2f} for {movieList[i]}')

filter=(movieList_df["number of ratings"] > 20)
movieList_df["pred"] = my_predictions
movieList_df = movieList_df.reindex(columns=["pred", "mean rating", "number of ratings", "title"])
movieList_df.loc[ix[:300]].loc[filter].sort_values("mean rating", ascending=False)