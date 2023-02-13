import matplotlib.pyplot as plt
import matplotlib as mpl
import tikzplotlib
import numpy as np

# Create some dummy data for 6 lines
x = np.linspace(0, 10, 10)
remapping = [0,0,0,0,0.625,0.8125,1,1,1,1]#np.random.rand(10)
different_constants = [0,0,0,0,0,0.6875,0.9375,1,1,1]
#both = [0,0,0,0,0,0.6875,0.9375,1,1,1]

print(len(remapping), len(different_constants))

# Plot the lines
plt.plot(x, remapping, label='done2-remapping')
plt.plot(x, different_constants, label='done2 with different constants')
#plt.plot(x, both, label='both')

# Add a legend, x-label, y-label, and grid
plt.legend(bbox_to_anchor=(1.03, 1), loc='upper left', borderaxespad=0.)

plt.xlabel('Fuel level')
plt.ylabel('Probability')
plt.grid()
# Show the plot
tikzplotlib.save("plots/taxi_remapping_comparison.tex")
