import numpy as np
import matplotlib.pyplot as plt
import tikzplotlib

# Create a 2D array of random values between 0 and 1
np.random.seed(0)
data = np.random.rand(4, 4)
# 0 Row
data[3][0] = 0.62
data[3][1] = 0.62
data[3][2] = 0.62
data[3][3] = 0.62

data[2][0] = 0.62
data[2][1] = 0
data[2][2] = 0.37
data[2][3] = 0

data[1][0] = 0.62
data[1][1] = 0.62
data[1][2] = 0.5
data[1][3] = 0

data[0][0] = 0
data[0][1] = 0.75
data[0][2] = 0.87
data[0][3] = 1

print(data)

# Create a figure and axes
fig, ax = plt.subplots()

# Set the colormap to be blue for low values and white for high values
cmap = plt.cm.get_cmap('Blues_r')
cmap.set_over('white')

# Plot the data using the custom colormap
c = ax.pcolormesh(data, cmap=cmap, vmax=1)

# Add a colorbar to the right of the plot
fig.colorbar(c, ax=ax, label='')


counter = 0
# Write the cell indices in each cell
for i in range(data.shape[0]):
    for j in range(data.shape[1]):
        ax.text(j+0.5, i+0.5, f'{counter}', ha='center', va='center')
        counter += 1


# Show graphic
tikzplotlib.save("plots/frozen_lake.tex")
