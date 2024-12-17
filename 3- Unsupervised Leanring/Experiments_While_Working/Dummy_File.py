import numpy as np
import matplotlib.pyplot as plt

def normal_distribution(x,mean,stand):
    out=-0.5*((x-mean)/stand)**2
    out= np.exp(out)
    out*= 1/(stand*2.5066)
    return out

x = np.linspace(0 ,10 , 100)
y = normal_distribution(x,5,3)
y_2= 10 * y
y_2=np.ceil(y_2)

print(y_2)
plt.hist(y_2, bins =100 )
plt.title(f"This is supposed to be Generate Gaussian Data")
plt.show()



