import math
import numpy as np
import matplotlib.pyplot as plt



def normal_distribution(x,mean,stand):
    out = np.array(len(x))
    out=-0.5*((x-mean)/stand)**2
    out= np.exp(out)
    out*= 1/(stand*2.5066)
    return out


def calculate_Outliers_IQR(x_distribution):
    # Making the Normal Distribution Using well-defined function
    # Calculating the higher and lower boundary for outliers
    q1 = np.quantile(x_distribution, 0.25)
    q2 = np.quantile(x_distribution, 0.5)
    q3 = np.quantile(x_distribution, 0.75)
    IQR = q3 - q1
    lower_outlier = q1 - 1.5 * IQR
    higher_outlier = q3 + 1.5 * IQR
    return lower_outlier,higher_outlier

def outliers_as_function_of_variance():
    size = 1000
    variance = 1
    mean = 50
    final_variance= 40

    x = np.linspace(0, 100, size)
    Percentage_Outlier =[]

    for current_variance in range(1, final_variance + 1):
        x_distribution = normal_distribution(x,mean,current_variance)
        lower_outlier, higher_outlier = calculate_Outliers_std(x_distribution, mean, current_variance)
        outlier_count, _ = count_outliers(x, lower_outlier, higher_outlier)
        outlier_percentage = (outlier_count / len(x)) * 100
        Percentage_Outlier.append(outlier_percentage)
    plt.plot(range(1, final_variance + 1), Percentage_Outlier, marker='o')
    plt.title(f"Percentage of Outliers When Changing Variance from 1 to {final_variance}")
    plt.xlabel("Variance")
    plt.ylabel("Percentage of Outliers")
    plt.grid(True)
    plt.show()


def calculate_Outliers_std(data, mean, std, k=2):
    """Calculate lower and upper outlier boundaries."""
    lower_Outlier = mean - k * std
    upper_Outlier = mean + k * std
    return lower_Outlier, upper_Outlier


def count_outliers(data, lower_bound, upper_bound):
    """Count the number of outliers in the data."""
    outliers = (data < lower_bound) | (data > upper_bound)
    return np.sum(outliers), outliers  # Return count and boolean mask
