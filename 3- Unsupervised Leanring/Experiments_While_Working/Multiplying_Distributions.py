import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from functions import normal_distribution
from matplotlib.widgets import Button



boundary = 20
size = 100

# Initial values for mean and standard deviation : We have X and Y
(x_initial_mean,x_initial_std) = (10,2)
(y_initial_mean,y_initial_std) = (10,2)

# Create a figure and axis
fig, ax = plt.subplots(1,3,figsize=(12, 6))
plt.subplots_adjust(left=0.1, bottom=0.3)  # Leave space for sliders


# Generate the initial distribution
x = np.linspace(0, boundary, size)
y = np.linspace(0,boundary,size)
x_distribution=normal_distribution(x,x_initial_mean,x_initial_std)
y_distribution=normal_distribution(y,y_initial_mean,y_initial_std)


# Plot the initial distribution

#Plotting X
line_x, = ax[0].plot(x, x_distribution, lw=2)
ax[0].set_title("Normal Distribution of X")
ax[0].set_xlabel("x")
ax[0].set_ylabel("Probability Density ")

#Plotting Y
line_y, = ax[2].plot(y, y_distribution, lw=2)
ax[2].set_title("Normal Distribution of Y")
ax[2].set_xlabel("y")
ax[2].set_ylabel("Probability Density")


# Add sliders for mean and standard deviation
ax_x_mean = plt.axes([0.1, 0.2, 0.35, 0.05])  # Position of mean slider
ax_x_std = plt.axes([0.1, 0.1, 0.35, 0.05])   # Position of std slider
slider_x_mean = Slider(ax_x_mean, 'Mean', 0, 100, valinit=x_initial_mean)
slider_x_std = Slider(ax_x_std, 'Std Dev', 0.1, 50, valinit=x_initial_std)



ax_y_mean = plt.axes([0.55, 0.2, 0.35, 0.05])  # Position of mean slider
ax_y_std = plt.axes([0.55, 0.1, 0.35, 0.05])   # Position of std slider
slider_y_mean = Slider(ax_y_mean, 'Mean', 0, 100, valinit=y_initial_mean)
slider_y_std = Slider(ax_y_std, 'Std Dev', 0.1, 50, valinit=y_initial_std)



# Update function for sliders
def update(val):
    new_mean_x = slider_x_mean.val
    new_std_x = slider_x_std.val
    new_mean_y = slider_y_mean.val
    new_std_y = slider_y_std.val

    # Update the distributions based on new slider values
    new_x_distribution = normal_distribution(x, new_mean_x, new_std_x)
    new_y_distribution = normal_distribution(y, new_mean_y, new_std_y)

    # Update the plots for X and Y distributions
    line_x.set_ydata(new_x_distribution)
    line_y.set_ydata(new_y_distribution)

    # Update the 3D surface based on the new distributions
    X, Y = np.meshgrid(new_x_distribution, new_y_distribution)
    Z = X * Y  # Update Z based on new X and Y distributions

    # Remove the old surface and plot the new one
    ax[1].cla()  # Clear the previous plot
    ax[1].plot_surface(X, Y, Z, cmap='viridis', edgecolor='none')  # Plot the new surface
    ax[1].set_title('3D Surface Plot of Cross Product')
    ax[1].set_xlabel('X Vector')
    ax[1].set_ylabel('Y Vector')
    ax[1].set_zlabel('Cross Product Value')

    # Redraw the canvas
    fig.canvas.draw_idle()

#Here we need to create the 2D data :

X, Y = np.meshgrid(x_distribution, y_distribution)
Z = X * Y

ax[1] = fig.add_subplot(1, 3, 2, projection='3d')
ax[1].plot_surface(X, Y, Z, cmap='viridis', edgecolor='none')
ax[1].set_title('3D Surface Plot of Cross Product')
ax[1].set_xlabel('X Vector')
ax[1].set_ylabel('Y Vector')
ax[1].set_zlabel('Cross Product Value')

# Connect the sliders to the update function
slider_x_mean.on_changed(update)
slider_x_std.on_changed(update)
slider_y_mean.on_changed(update)
slider_y_std.on_changed(update)

plt.show()