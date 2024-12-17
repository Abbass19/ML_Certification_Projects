import  numpy as np
from Making_Normal_Distributions import *
import matplotlib.pyplot as plt


def Error(data, mu , std):
    error_mu = (mu - np.mean(data))/mu
    error_std = (std - np.std(data))/mu
    return error_mu,error_std


def Optimizing_Nb_Divisions(limit_divisions,data_points,plot):

    iterations = limit_divisions
    error = np.zeros((iterations,2))
    for nb_divisions in range(0,iterations,1):
        DATA = Generate_gaussian_sample_testing_divisions(12,4,data_points,0,25,0,nb_divisions)
        error[nb_divisions] = np.array(Error(DATA,12,4))
        print(f"This is iterator {nb_divisions}  with error error {error[nb_divisions]}")

    #Find the minimum Sum
    #Adding Both errors and Finding the local minimum
    tracking = error[:,0] + error[:,1]
    tracking = tracking[15:]
    index_of_lowest = np.argmin(tracking) +15
    print(f"The value with least error is {index_of_lowest}")

    if plot:
        #Plotting Graph
        plt.plot(error[:, 0], label='Percentage of Mean Error')
        plt.plot(error[:, 1], label=' Percentage of Standard Deviation Error ')
        plt.axvline(x=index_of_lowest, color='r', linestyle='--', label='Minimum Error Line')
        plt.legend()
        plt.show()

    return  data_points, error[index_of_lowest]


def Optimizing_Data_points(limit_points,step):
    #Becasue the function that generates sample is crazy we will start at 300

    limit_divisions = 500
    steps = (limit_points-300)/step
    steps = int(np.floor(steps))
    Result = []

    for data_points in range(300,limit_points,100):
        iterator = int(data_points/100)
        Result.append(Optimizing_Nb_Divisions(limit_divisions,data_points,0))

    data_points = [item[0] for item in Result]
    best_errors = [item[1] for item in Result]
    plt.plot(data_points, best_errors, marker='o', linestyle='-')
    plt.title("Best Error Index vs. Data Points")
    plt.xlabel("Data Points")
    plt.ylabel("Error at Lowest Index")
    plt.grid(True)

def Standard_Function_Comparison(mu,std,num_points):

    data_customized = Generate_gaussian_sample(mu,std,num_points,0,25,0)
    customized_mu_error , customized_std_error = Error(data_customized,mu,std)
    data_standard = np.random.normal(loc=mu, scale=std, size=num_points)
    standard_mu_error , standard_std_error = Error(data_standard,mu,std)

    print(f" Customized data had mu error of {customized_mu_error} and std error of {customized_std_error}")
    print(f" Standard data had mu error of {standard_mu_error} and std error of {standard_std_error}")
    print(f"The ratio of mu error is {customized_mu_error/standard_mu_error}")
    print(f"The ratio of the std error is {customized_std_error/standard_std_error}")






Optimizing_Nb_Divisions(500,1000,1)

Optimizing_Data_points(2500,100)

# data = Generate_gaussian_sample(4,2,1000,0,24,1)
Standard_Function_Comparison(4,2,1000)