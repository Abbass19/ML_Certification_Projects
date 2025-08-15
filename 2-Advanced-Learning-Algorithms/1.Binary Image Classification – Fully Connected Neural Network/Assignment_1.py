import numpy as np
import warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt
from autils import *
import logging
import warnings
from functions import imagePlotter,imagePlotterComparator
logging.getLogger("tensorflow").setLevel(logging.ERROR)
tf.autograph.set_verbosity(0)
warnings.simplefilter(action='ignore', category=FutureWarning)



#Loading Data
X, y = load_data()
print(f"The shape of x is {X.shape}")

#Plot Images with ground truth label
# imagePlotter(X,y)

#Build a Model using TensorFlow Sequential Library
model = Sequential(
    [
        tf.keras.Input(shape=(400,)),
        Dense(units = 25, activation = 'sigmoid', name = 'layer_1'),
        Dense(units = 15, activation = 'sigmoid' ,name = 'later_2'),
        Dense(units  = 1, activation = 'sigmoid', name = 'layer_3')
    ], name = "my_model"
)
model.summary()


# Unpacking layers to freeze some and maybe apply transfer learning
[layer1, layer2, layer3] = model.layers
print(model.layers[2].weights)
W1,b1 = layer1.get_weights()
W2,b2 = layer2.get_weights()
W3,b3 = layer3.get_weights()
print(f"W1 shape = {W1.shape}, b1 shape = {b1.shape}")
print(f"W2 shape = {W2.shape}, b2 shape = {b2.shape}")
print(f"W3 shape = {W3.shape}, b3 shape = {b3.shape}")


# Compile and Fit using TensorFlow library
model.compile(
    loss=tf.keras.losses.BinaryCrossentropy(),
    optimizer=tf.keras.optimizers.Adam(0.001),
)
model.fit(
    X,y,
    epochs=20
)


#Function to compare ground truth and prediction of random elements
# imagePlotterComparator(X,y,model)


#Now we need to use the stuff we are have built:





