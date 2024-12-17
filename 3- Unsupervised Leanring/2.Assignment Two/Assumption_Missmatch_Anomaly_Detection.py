import os
import time
import numpy as np
import matplotlib.pyplot as plt

# I want to make this general for more than one assumption

def normal_distribution(x,mean,stand):
    out=-0.5*((x-mean)/stand)**2
    out= np.exp(out)
    out*= 1/(stand*2.5066)
    return out

def exponential_distribution(x_axis,y):
    return  y * np.exp(-y*x_axis)




def estimate_gaussian(X):

    """
    This function just finds mean and standard deviation
    """
    m, n = X.shape
    mu = np.zeros(n)
    var = np.zeros(n)
    for i in range(n):
        mu[i] = np.mean(X[:, i])
        var[i] = np.var(X[:, i])
    return mu, var


def multivariate_gaussian(X, mu, var):
    """
    Computes the probability density of each feature (column) independently for the multivariate Gaussian distribution.
    Each column has its own mean (mu) and variance (var).
    """
    # Ensure inputs are numpy arrays
    X = np.asarray(X)
    mu = np.asarray(mu)
    var = np.asarray(var)

    # Broadcast shapes for compatibility
    assert X.shape[1] == len(mu) == len(var), "Mismatch between X columns, mu, and var dimensions."

    # Compute prefactor and exponent for each column independently
    prefactor = 1 / np.sqrt(2 * np.pi * var)
    exponent = -0.5 * ((X - mu) ** 2 / var)

    # Compute probability densities for each column
    p = prefactor * np.exp(exponent)

    return p


def multivariate_exponential(X, lambdas):
    X = np.asarray(X)
    assert X.shape[1] == len(lambdas), "Mismatch between dimensions of X and lambdas"
    if np.any(X < 0):
        raise ValueError("All values in X must be non-negative for the exponential distribution.")
    densities = lambdas * np.exp(-lambdas * X)
    return densities



def select_threshold(y_val, p_val):
    """
    This function iterates over multiple epsilon and chooses the one that leads
    to the highest F1 score.
    """
    best_epsilon = 0
    best_F1 = 0

    step_size = (max(p_val) - min(p_val)) / 1000

    for epsilon in np.arange(min(p_val), max(p_val), step_size):

        predictions = (p_val < epsilon)
        false_positive = sum((predictions == 1) & (y_val == 0))
        true_positive = np.sum((predictions == 1) & (y_val == 1))
        false_negative = np.sum((predictions == 0) & (y_val == 1))

        precision = true_positive / (true_positive + false_positive+0.01)
        recall = true_positive / (true_positive + false_negative+0.01)
        F1 = (2*precision*recall)/(precision + recall+0.01)

        if F1 > best_F1:
            best_F1 = F1
            best_epsilon = epsilon
    print(f"The best epsilon is {best_epsilon}")
    return best_epsilon, best_F1


def anomaly_detection(X_train, X_val, y_val):
    mu, var = estimate_gaussian(X_train)
    p = multivariate_gaussian(np.copy(X_train), mu, var)
    p_val = multivariate_gaussian(np.copy(X_val), mu, var)
    p_val = np.prod(p_val, axis = 1 )
    epsilon, F1 = select_threshold(y_val, p_val)
    p = np.prod(p,axis = 1)
    outliers = p < epsilon
    percentage_outliers = np.mean(outliers)
    return percentage_outliers


def test_assumption_mismatch(anomaly_detection_function):
    #This function takes an assumption that data is normal
    #Here we make a procedure to check what can this result

    #Making x array
    X = np.linspace(0, 25, 307)
    X = np.column_stack((X,X))

    #Choosing Initial Epsilon
    epsilon_normal = 0.000003
    epsilon_exponential =0.1


    #Making Pure Exponential Data
    lambdas = np.array([1.5 ,1])
    X_main_exp = multivariate_exponential(X,lambdas)

    X_shuffled_exp = np.copy(X_main_exp)  # Create a copy to avoid modifying the original array
    for col in range(X_main_exp.shape[1]):
        np.random.shuffle(X_shuffled_exp[:, col])

    y_main_exp = np.prod(X_shuffled_exp ,axis=1)
    y_main_exp = (y_main_exp < epsilon_exponential)
    PO_exp = np.mean(y_main_exp)



    #Splitting Dataset : Preparation for function
    X_train_exp = X_shuffled_exp[::2]
    X_valid_exp = X_shuffled_exp[1::2]
    y_valid_exp = y_main_exp[1::2]


    #Making Pure Normalized Data:
    mu = np.array([3, 3])
    var = np.array([1, 1])
    X_main_normal = multivariate_gaussian(X,mu,var)
    X_shuffled_normal = np.copy(X_main_normal)  # Create a copy to avoid modifying the original array
    for col in range(X_main_normal.shape[1]):
        np.random.shuffle(X_shuffled_normal[:, col])


    y_main_normal = np.prod(X_shuffled_normal, axis=1)
    y_main_normal = (y_main_normal<epsilon_normal)
    PO_normal = np.mean(y_main_normal)


    #Splitting Dataset : Preparation for function
    X_train_normal = X_shuffled_normal[::2]
    X_valid_normal = X_shuffled_normal[1::2]
    y_valid_normal =y_main_normal[1::2]
    X = np.arange(0, 10, 0.01629)
    x_axis = X[::2]


    fig , axis = plt.subplots(1,2)

    axis[0].scatter(X_valid_exp[:,0], X_valid_exp[:, 1])
    axis[0].set_title(" X_valid_exp  data")
    Outliers = X_valid_exp[y_valid_normal==1]
    axis[0].scatter(Outliers[:,0],Outliers[:,1],color='red')
    axis[0].set_title(" Outliers of Exponential  data")

    axis[1].scatter(X_main_normal[:, 0], X_main_normal[:, 1])
    axis[1].set_title(" X_main_normal  data")
    # Outliers = X_main_normal[y_valid_normal == 1]
    # axis[1].scatter(Outliers[:, 0], Outliers[:, 1], color='red')
    # axis[1].set_title(" Outliers of X_valid_normal  data")

    plt.show()


    # #Finding the right epsilon : Iterate till PO initial is the same
    # while PO_normal <0.1 or PO_normal>0.2:
    #     print(f"The PO_normal is {PO_normal}")
    #
    #     if PO_normal<0.1:
    #         epsilon_normal *= 1
    #     if PO_normal>0.2:
    #         epsilon_normal *= 1
    #     Y_main_normal = multivariate_gaussian(X_main_normal,mu,var)
    #     y_main_normal = (Y_main_normal<epsilon_normal)
    #     PO_normal = np.mean(y_main_normal)

    # print(f"The shape of X_train_exp is {X_train_exp.shape}")
    # print(f"The shape of X_valid_exp is {X_valid_exp.shape}")
    # print(f"The shape of y_valid_exp is {y_valid_exp.shape}")


    #Now we make our Testing :
    PO_exp_after    = anomaly_detection_function(X_train_exp,X_valid_exp,y_valid_exp)
    PO_normal_after = anomaly_detection_function(X_train_normal,X_valid_normal,y_valid_normal)


    #How much did the PO change
    print(f"The PO of normal data is {PO_normal} and was approximated as {PO_normal_after}")
    print(f"The PO of exponential was {PO_exp} and was approximated as {PO_exp_after}")




def load_data():
    X = np.load("data/X_part1.npy")
    X_val = np.load("data/X_val_part1.npy")
    y_val = np.load("data/y_val_part1.npy")
    return X, X_val, y_val

def load_data_multi():
    X = np.load("data/X_part2.npy")
    X_val = np.load("data/X_val_part2.npy")
    y_val = np.load("data/y_val_part2.npy")
    return X, X_val, y_val



def visualize_fit(X, mu, var):

    """
    This visualization shows you the
    probability density function of the Gaussian distribution. Each example
    has a location (x1, x2) that depends on its feature values.
    """

    X1, X2 = np.meshgrid(np.arange(0, 35.5, 0.5), np.arange(0, 35.5, 0.5))
    Z = multivariate_gaussian(np.stack([X1.ravel(), X2.ravel()], axis=1), mu, var)
    Z = Z.reshape(X1.shape)

    plt.plot(X[:, 0], X[:, 1], 'bx')

    if np.sum(np.isinf(Z)) == 0:
        plt.contour(X1, X2, Z, levels=10 ** (np.arange(-20., 1, 3)), linewidths=1)

    # Set the title
    plt.title("The Gaussian contours of the distribution fit to the dataset")
    # Set the y-axis label
    plt.ylabel('Throughput (mb/s)')
    # Set the x-axis label
    plt.xlabel('Latency (ms)')
    plt.show()





