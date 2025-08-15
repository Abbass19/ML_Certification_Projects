import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.activations import linear, relu, sigmoid
import matplotlib.pyplot as plt
import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)
tf.autograph.set_verbosity(0)
from autils import *
from lab_utils_softmax import plt_softmax
from functions import imagePlotter
np.set_printoptions(precision=2)
from functions import my_softmax

# load dataset
X = np.load('data/X.npy')
y = np.load('data/y.npy')
print(f"The shape of X is {X.shape}\n")
print(f"The shape of y is {y.shape}")

#Fuction that takes some images and shows their annotations

# imagePlotter(X,y)



# Sequential model
tf.random.set_seed(1234) # for consistent results
model = Sequential(
    [
        tf.keras.layers.InputLayer(input_shape=(400,)),
        Dense(25,activation='relu',name='L1'),
        Dense(15,activation='relu',name='L2'),
        Dense(10,activation='linear',name = 'L3')
    ], name = "my_model"
)
model.summary()

#Extracting Layers
[layer1, layer2, layer3] = model.layers
W1,b1 = layer1.get_weights()
W2,b2 = layer2.get_weights()
W3,b3 = layer3.get_weights()
print(f"W1 shape = {W1.shape}, b1 shape = {b1.shape}")
print(f"W2 shape = {W2.shape}, b2 shape = {b2.shape}")
print(f"W3 shape = {W3.shape}, b3 shape = {b3.shape}")


#Training the model
model.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
)

history = model.fit(
    X,y,
    epochs=40
)

plot_loss_tf(history)

imagePlotterComparator(X,y,model)
