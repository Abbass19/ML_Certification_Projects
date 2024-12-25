import numpy as np
import numpy.ma as ma
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import tabulate
from recsysNN_utils import *

import matplotlib.pyplot as plt



top10_df = pd.read_csv("./data/content_top10_df.csv")
by_genre_df = pd.read_csv("./data/content_bygenre_df.csv")



# Load Data, set configuration variables
item_train, user_train, y_train, item_features, user_features, item_vecs, movie_dict, user_to_genre = load_data()
"""
    1. item_train : A file that has movies identified with ID of a movie and contain the rating of the movie, 
    number of people who rated it, and 14 features that are engineered. Has 50884 records. movie id,year,ave rating,
    Action,Adventure,Animation,Children,Comedy,Crime,Documentary,Drama,Fantasy,Horror,Mystery,Romance,Sci-Fi,Thriller

        This file is crazy. All the records with the same ID are the same
        
        
    2. user_train : A file that has user identified with ID con. It describes the user with 14 numeric features
     from 0 to 5. They are Action,Adventure,Animation,Children,Comedy,Crime,Documentary,Drama,Fantasy,Horror,
     Mystery,Romance,Sci-Fi,Thriller
        This is also crazy because it has 70 records the same. I don't know what does this imply 
        
    3. y_train has 1 column and 5084 records : What does this represent. Not my user
    
    4. Item_features : item_train_header : movie id,year,ave rating,Action,Adventure,Animation,Children,Comedy,
    Crime,Documentary,Drama,Fantasy,Horror,Mystery,Romance,Sci-Fi,Thriller
    
    5.User_features : user id,rating count,rating ave,Action,Adventure,Animation,Children,Comedy,Crime,
    Documentary,Drama,Fantasy,Horror,Mystery,Romance,Sci-Fi,Thriller
    
    6. Item_vecs : The data set of movies with 14 features for each one. Basically we take this one, and multiply
    each record by different numbers and we get 
    
    7. movie_dict : Is a dictionary datastructures, that has movie ID, movie name, and the label(Action/Adventure) 
    
    8. user_to_genre (I don't know what is this file. It uses pickle and it is not working)
    
    Now we understood the situation, We have only one user. The rating for 14 features for all movies 
    is already done. In contrast to the situation in Collaborative Filtering Algorithm. 

"""
num_user_features = user_train.shape[1] - 3  # remove userid, rating count and ave rating during training
num_item_features = item_train.shape[1] - 1  # remove movie id at train time

print(f"The number of features for user {num_user_features}")
print(f"The number of features for a movie are {num_item_features}")


uvs = 3  # user genre vector start
ivs = 3  # item genre vector start
u_s = 3  # start of columns to use in training, user
i_s = 1  # start of columns to use in training, items
print(f"Number of training vectors: {len(item_train)}")


print(f"Understanding Variables before Scaling")
print(f"We have 3 things to discover : item_train, user_train, y_train ")
print(f"The shape of item_train {item_train.shape}")
print(f"The shape of user_train {user_train.shape}")
print(f"The shape of y_train {y_train.shape}")



# scale training data
item_train_unscaled = item_train
user_train_unscaled = user_train
y_train_unscaled    = y_train

scalerItem = StandardScaler()
scalerItem.fit(item_train)
item_train = scalerItem.transform(item_train)

scalerUser = StandardScaler()
scalerUser.fit(user_train)
user_train = scalerUser.transform(user_train)

scalerTarget = MinMaxScaler((-1, 1))
scalerTarget.fit(y_train.reshape(-1, 1))
y_train = scalerTarget.transform(y_train.reshape(-1, 1))
#ynorm_test = scalerTarget.transform(y_test.reshape(-1, 1))




#We want to compare each data at feature 5, before and after normalization
#We want to see the feature number 5  :


for i in range(5):

    explore_feature = 3+i
    fig , axis = plt.subplots(1,2)

    axis[0].hist(item_train_unscaled[:,explore_feature])
    axis[0].set_title("Item Train Before Normalization")
    axis[1].hist(item_train[:,explore_feature])
    axis[1].set_title("Item Train After Normalization ")


    axis[0].hist(user_train_unscaled[:,explore_feature])
    axis[0].set_title("User Train Before Normalization")
    axis[1].hist(user_train[:,explore_feature])
    axis[1].set_title("User Train After Normalization ")


    axis[0].hist(y_train_unscaled)
    axis[0].set_title("Y Train Before Normalization")
    axis[1].hist(y_train)
    axis[1].set_title("Y Train After Normalization ")
    # plt.show()


item_train, item_test = train_test_split(item_train, train_size=0.80, shuffle=True, random_state=1)
user_train, user_test = train_test_split(user_train, train_size=0.80, shuffle=True, random_state=1)
y_train, y_test       = train_test_split(y_train,    train_size=0.80, shuffle=True, random_state=1)
print(f"movie/item training data shape: {item_train.shape}")
print(f"movie/item test data shape: {item_test.shape}")


