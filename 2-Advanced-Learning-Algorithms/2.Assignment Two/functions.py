import numpy as np
import matplotlib.pyplot as plt



def my_softmax(z):
    z = np.exp(z)
    a = z / sum(z)
    return a


#This is the same function that we used in assignment 1 for viewing
# random set of images with the ground truth label for them
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
