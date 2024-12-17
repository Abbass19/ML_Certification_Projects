import numpy as np
import matplotlib.pyplot as plt
import statistics


def calculate_normal_probability(x, mean, stand):
    out=-0.5*((x-mean)/stand)**2
    out= np.exp(out)
    out*= 1/(stand*2.5066)
    return out

def threshold(number):
      delta = ((np.ceil(number) - number) - (number - np.floor(number)))
      if delta>0 :
          return np.floor(number)
      return np.ceil(number)
def Generate_gaussian_sample(mu,std,nb_points,Start,End, Informative):

    """"
    Customized function that generates the Gaussian data the same as
    np.random.normal
    The Information is binary variable, if it is one it shows the error graphs
    """


    #This formula is deduced from analyzing the testing functions
    #Results are in the Excel file named "Best index vs data points" in the directory
    nb_divisions = np.floor(nb_points/20)+40
    nb_divisions = int(nb_divisions)

    #Data Manipulation
    DATA = []
    x = np.linspace(Start,End,nb_divisions+1)
    center = (x + np.roll(x, -1))/2
    probability = calculate_normal_probability(center, mu, std)
    Probability_sum = sum(probability)

    for i in range(len(x)-1):
        tokens = nb_points - len(DATA)
        if Informative :
            print(f"The number of token is {tokens}")
        step = probability[i]*nb_points/Probability_sum
        step=int(threshold(step))
        if step>tokens:
            local_data = np.linspace(x[i], x[i + 1], tokens)
            DATA.extend(local_data.tolist())
            DATA_numpy = np.array(DATA)
            continue
        local_data = np.linspace(x[i],x[i+1],int(step))
        if Informative :
            print(f"The nb is points for this sector : {int(np.ceil(probability[i]*nb_points))} and it took {len(local_data)}")
        DATA.extend(local_data.tolist())


    #Filling the empty spaces with the Mean
    DATA_numpy = np.array(DATA)
    difference = nb_points - len(DATA_numpy)
    difference = np.ones(difference) * np.mean(DATA_numpy)
    DATA_numpy = np.append(DATA_numpy , difference)

    #Calcuate the error for mean and standard
    DATA_Ceil = np.ceil(DATA_numpy)
    DATA_Floor = np.floor(DATA_numpy)
    Error_Ceil = abs(np.mean(DATA_Ceil) - mu) + abs(np.std(DATA_Ceil) - std)
    Error_Floor = abs(np.mean(DATA_Floor) - mu) + abs(np.std(DATA_Floor) - std)


    #Returing Part
    if Error_Ceil < Error_Floor :
         data = DATA_Ceil
    data=  DATA_Floor


    #Draing Error Probability
    if Informative:
        print(f"The Length of Data is {len(DATA_numpy)}")
        sample_mean = sum(DATA_numpy) / len(DATA_numpy)
        sample_deviation = np.std(DATA_numpy)
        print(f"The Length of Data is {len(DATA_numpy)}")
        x_axis = np.linspace(Start, End, nb_points * 100)
        fig , axis = plt.subplots(1,2)
        axis[0].scatter(x_axis,calculate_normal_probability(x_axis,mu,std))
        axis[0].set_title(f"The Original Normal with Mean {mu} and Variance {std}")
        axis[1].hist(DATA,bins=int(np.ceil(nb_points/10)))
        axis[1].set_title(f"Sample Data having Mean {sample_mean} and Variance {sample_deviation}")
        plt.show()
        print(f"The Length of Data is {len(DATA_numpy)}")

    return  data


def Generate_gaussian_sample_testing_divisions(mu,std,nb_points,Start,End, Informative,nb_divisions):

    """
    This function has only one hyperparameter as an outside parameter so that we can
    make the testing in the Optimization Procedure page
    """

    #Data Manipulation
    DATA = []
    x = np.linspace(Start,End,nb_divisions+1)
    center = (x + np.roll(x, -1))/2
    probability = calculate_normal_probability(center, mu, std)
    Probability_sum = sum(probability)

    for i in range(len(x)-1):
        tokens = nb_points - len(DATA)
        if Informative :
            print(f"The number of token is {tokens}")
        step = probability[i]*nb_points/Probability_sum
        step=int(threshold(step))
        if step>tokens:
            local_data = np.linspace(x[i], x[i + 1], tokens)
            DATA.extend(local_data.tolist())
            DATA_numpy = np.array(DATA)
            continue
        local_data = np.linspace(x[i],x[i+1],int(step))
        if Informative :
            print(f"The nb is points for this sector : {int(np.ceil(probability[i]*nb_points))} and it took {len(local_data)}")
        DATA.extend(local_data.tolist())


    #Filling the empty spaces with the Mean
    DATA_numpy = np.array(DATA)
    difference = nb_points - len(DATA_numpy)
    difference = np.ones(difference) * np.mean(DATA_numpy)
    DATA_numpy = np.append(DATA_numpy , difference)

    #Calcuate the error for mean and standard
    DATA_Ceil = np.ceil(DATA_numpy)
    DATA_Floor = np.floor(DATA_numpy)
    Error_Ceil = abs(np.mean(DATA_Ceil) - mu) + abs(np.std(DATA_Ceil) - std)
    Error_Floor = abs(np.mean(DATA_Floor) - mu) + abs(np.std(DATA_Floor) - std)


    #Returing Part
    if Error_Ceil < Error_Floor :
         data = DATA_Ceil
    data=  DATA_Floor


    #Draing Error Probability
    if Informative:
        print(f"The Length of Data is {len(DATA_numpy)}")
        sample_mean = sum(DATA_numpy) / len(DATA_numpy)
        sample_deviation = np.std(DATA_numpy)
        print(f"The Length of Data is {len(DATA_numpy)}")
        x_axis = np.linspace(Start, End, nb_points * 100)
        fig , axis = plt.subplots(1,2)
        axis[0].scatter(x_axis,calculate_normal_probability(x_axis,mu,std))
        axis[0].set_title(f"The Original Normal with Mean {mu} and Variance {std}")
        axis[1].hist(DATA,bins=int(np.ceil(nb_points/10)))
        axis[1].set_title(f"Sample Data having Mean {sample_mean} and Variance {sample_deviation}")
        plt.show()
        print(f"The Length of Data is {len(DATA_numpy)}")

    return  data

























