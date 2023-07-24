import pandas as pd
import matplotlib.pyplot as plt
from io import StringIO
import numpy as np

FONT_SIZE=18

plt.xticks(fontsize=FONT_SIZE)  # Adjust to desired font size
plt.yticks(fontsize=FONT_SIZE)  # Adjust to desired font size

df = pd.read_csv("rl_model_checking/all_pruning.txt", names=["pruning", "label", "percentage", "unknown", "safety_measure"])
df['id'] = df['pruning'].astype(str) + "_" + df['label'].astype(str)
# Create the 'id' column after replacement
#df['id'] = df['label'].astype(str) + df['unknown'].astype(str)

# Remove 'min' from 'id' string
#df['id'] = df['id'].str.replace('min', '')

labels = df['id'].unique()

for label in labels:
    subset = df[df['id'] == label]
    if label.find("jobs_done=2") != -1:
        if label.find("random") != -1:
            label = "random (jobs=2)"
            plt.plot(subset['percentage'], subset['safety_measure'],  '-x', label=label)
        else:
            label = "L1 (jobs=2)"
            plt.plot(subset['percentage'], subset['safety_measure'],  '-o', label=label)
    elif label.find("jobs_done=1") != -1:
        if label.find("random") != -1:
            label = "random (jobs=1)"
            plt.plot(subset['percentage'], subset['safety_measure'],  '-x', label=label)
        else:
            label = "L1 (jobs=1)"
            plt.plot(subset['percentage'], subset['safety_measure'],  '-o', label=label)
    elif label.find("empty") != -1:
        if label.find("random") != -1:
            label = "random (empty)"
            plt.plot(subset['percentage'], subset['safety_measure'],  '-x', label=label)
        else:
            label = "L1 (empty)"
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


print(df.head())
# save figure to png
plt.savefig('different_prunings.png', dpi=300, bbox_inches='tight')
plt.savefig('different_prunings.eps', dpi=300, bbox_inches='tight', format='eps')


