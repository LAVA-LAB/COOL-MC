import matplotlib.pyplot as plt
import matplotlib as mpl
import tikzplotlib
import numpy as np

# Create some dummy data for 6 lines
x = np.linspace(0, 10, 10)
jobs2 = [0,0,0,0,0.75,0.9375,1,1,1,1]
jobs1 = [0,0,1,1,1,1,1,1,1,1]
empty = [1,1,1,1,0,0,0,0,0,0]



# Plot the lines
plt.plot(x, jobs2, label='done2')
plt.plot(x, jobs1, label='done1')
plt.plot(x, empty, label='empty')

# Add a legend, x-label, y-label, and grid
plt.legend(bbox_to_anchor=(1.03, 1), loc='upper left', borderaxespad=0.)

plt.xlabel('Fuel level')
plt.ylabel('Probability')
plt.grid()
# Show the plot
tikzplotlib.save("plots/taxi_abstractions.tex")
