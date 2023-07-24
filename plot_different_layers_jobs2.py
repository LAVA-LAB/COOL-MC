import pandas as pd
import matplotlib.pyplot as plt
from io import StringIO
import numpy as np

FONT_SIZE=18

plt.xticks(fontsize=FONT_SIZE)  # Adjust to desired font size
plt.yticks(fontsize=FONT_SIZE)  # Adjust to desired font size

df = pd.read_csv("data.csv", names=["label", "percentage", "unknown", "safety_measure"])
df['id'] = df['label'].astype(str) + df['unknown'].astype(str)
# Replace values in 'unknown' column
df['unknown'] = df['unknown'].replace({-5: ' (first)', -4: ' (second)', -3: ' (third)', -2: ' (fourth)', -1: ' (fifth)'})

# Create the 'id' column after replacement
df['id'] = df['label'].astype(str) + df['unknown'].astype(str)

# Remove 'min' from 'id' string
df['id'] = df['id'].str.replace('min', '')

labels = df['id'].unique()

for label in labels:
    subset = df[df['id'] == label]
    if label.find("jobs_done=2") != -1:
        if label.find("(fifth)") != -1:
            label = "jobs=2 (fifth)"
            plt.plot(subset['percentage'], subset['safety_measure'],  '-x', label=label)
        elif label.find("(fourth)") != -1:
            label = "jobs=2 (fourth)"
            plt.plot(subset['percentage'], subset['safety_measure'],  '-|', label=label)
        elif label.find("(third)") != -1:
            label = "jobs=2 (third)"
            plt.plot(subset['percentage'], subset['safety_measure'],  '-s', label=label)
        elif label.find("(second)") != -1:
            label = "jobs=2 (second)"
            plt.plot(subset['percentage'], subset['safety_measure'],  '-*', label=label)
        else:
            label = "jobs=2 (first)"
            plt.plot(subset['percentage'], subset['safety_measure'],  '-o', label=label)



# Set the tick locators for the x and y axes to multiples of 0.05
plt.xticks(np.arange(0, 1+0.05, 0.1))
plt.yticks(np.arange(0, 1+0.05, 0.1))

# Enable the grid and set alpha to 0.2
plt.grid(True, alpha=0.2)
plt.xlabel('Percentage', fontsize=FONT_SIZE)
plt.ylabel('Safety Measure', fontsize=FONT_SIZE)

# Move the legend outside of the plot at the upper right corner
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=FONT_SIZE)

# save figure to png
plt.savefig('plot_jobs2.png', dpi=300, bbox_inches='tight')
plt.savefig('plot_jobs2.eps', dpi=300, bbox_inches='tight', format='eps')


