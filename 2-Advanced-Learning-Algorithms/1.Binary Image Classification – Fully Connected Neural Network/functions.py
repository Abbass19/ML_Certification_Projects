import matplotlib.pyplot as plt
import numpy as np

#This class is specified for Analyzing and understanding all the functions used and
#to be able to set them apart.


def imagePlotter(X,y):
    rows, columns = X.shape
    fig, axes = plt.subplots(8, 8, figsize=(8, 8))
    fig.tight_layout(pad=0.1)
    for iterator, ax in enumerate(axes.flat):
        random_index = np.random.randint(rows)
        X_random_reshaped = X[random_index].reshape((20, 20)).T
        ax.imshow(X_random_reshaped, cmap='gray')
        ax.set_title(y[random_index, 0])
        ax.set_axis_off()
    plt.show()



def imagePlotterComparator(X, y, model):
    rows, columns = X.shape
    fig, axes = plt.subplots(8, 8, figsize=(8, 8))
    fig.tight_layout(pad=0.1, rect=[0, 0.03, 1, 0.92])  # [left, bottom, right, top]
    for i, ax in enumerate(axes.flat):
        random_index = np.random.randint(rows)
        X_random_reshaped = X[random_index].reshape((20, 20)).T
        ax.imshow(X_random_reshaped, cmap='gray')
        prediction = model.predict(X[random_index].reshape(1, 400))
        yhat = threshold(prediction)
        ax.set_title(f"{y[random_index, 0]},{yhat}")
        ax.set_axis_off()
    fig.suptitle("Label, yhat", fontsize=16)
    plt.show()


def threshold(prediction):
    if prediction >= 0.5:
       return  1
    else:
        return 0


def my_dense(a_in, W, b, g):
    """
        Our function should look like this
        my_dense(units = 25, activation = 'sigmoid', name = 'layer_1'),
        """
    units = W.shape[1]
    a_out = np.zeros(units)
    for i in range(units):
        z=np.dot(W[:,i],a_in) + b[i]
        a_out[i]=g(z)
    return a_out

def my_sequential(x, W1, b1, W2, b2, W3, b3):
    """
    This is more rigid than the code below :
         [
        tf.keras.Input(shape=(400,)),
        Dense(units = 25, activation = 'sigmoid', name = 'layer_1'),
        Dense(units = 15, activation = 'sigmoid' ,name = 'later_2'),
        Dense(units  = 1, activation = 'sigmoid', name = 'layer_3')
            ],
     name = "my_model"
         """
    a1 = my_dense(x,  W1, b1, sigmoid)
    a2 = my_dense(a1, W2, b2, sigmoid)
    a3 = my_dense(a2, W3, b3, sigmoid)
    return a3


