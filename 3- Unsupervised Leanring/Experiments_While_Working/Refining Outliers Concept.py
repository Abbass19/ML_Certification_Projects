import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from functions import normal_distribution,calculate_Outliers_std,count_outliers,outliers_as_function_of_variance


# Function to update the plot
def update(val):
    # Get updated mean and standard deviation from sliders
    new_mean = Slider_1.val
    new_std = Slider_2.val

    # Update the normal distribution
    new_x_distribution = normal_distribution(x, new_mean, new_std)
    line_x.set_ydata(new_x_distribution)


    # Calculate new outlier boundaries
    new_lower_outlier, new_higher_outlier = calculate_Outliers_std(new_x_distribution, new_mean, new_std)

    # Count outliers
    outlier_count, _ = count_outliers(x, new_lower_outlier, new_higher_outlier)
    outlier_percentage = (outlier_count / len(x)) * 100

    # Update vertical lines' positions
    line_1.set_xdata([new_lower_outlier])
    line_2.set_xdata([new_higher_outlier])

    # Add the text box for the outlier percentage
    text_box = ax.text(0.05, 0.9, f"Outliers: {outlier_percentage:.2f}%", transform=ax.transAxes,
                       fontsize=12, bbox=dict(facecolor='white', alpha=0.8))

    # Redraw the figure to show updates
    fig.canvas.draw_idle()


# Initial parameters for normal distribution
mean, std = 50, 10
size = 1000
x = np.linspace(0, 100, size)
x_distribution = normal_distribution(x, mean, std)

# Calculate initial outlier boundaries
lower_outlier, higher_outlier = calculate_Outliers_std(x_distribution, mean, std)

# Set up the figure and plot
fig, ax = plt.subplots()
plt.subplots_adjust(left=0.1, bottom=0.3)
line_x, = plt.plot(x, x_distribution, label="Normal Distribution")

# Add vertical lines for outliers
line_1 = plt.axvline(x=lower_outlier, color='r', linestyle='--', label="Lower Outlier")
line_2 = plt.axvline(x=higher_outlier, color='r', linestyle='--', label="Higher Outlier")

# Set up sliders
slider_mean_ax = plt.axes([0.4, 0.2, 0.35, 0.05])
slider_std_ax = plt.axes([0.4, 0.1, 0.35, 0.05])
Slider_1 = Slider(slider_mean_ax, 'Mean', 0, 100, valinit=mean)
Slider_2 = Slider(slider_std_ax, 'Std Dev', 1, 50, valinit=std)

# Connect sliders to the update function
Slider_1.on_changed(update)
Slider_2.on_changed(update)

# Finalize plot
plt.legend()
plt.show()


outliers_as_function_of_variance()