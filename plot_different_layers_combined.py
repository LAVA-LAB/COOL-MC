import pandas as pd
import matplotlib.pyplot as plt
from io import StringIO
import numpy as np

df = pd.read_csv("data.csv", names=["label", "percentage", "unknown", "safety_measure"])
df['id'] = df['label'].astype(str) + df['unknown'].astype(str)
# Replace values in 'unknown' column
df['unknown'] = df['unknown'].replace({-5: ' (first)', -4: ' (second)', -3: ' (third)', -2: ' (fourth)', -1: ' (fifth)'})

# Create the 'id' column after replacement
df['id'] = df['label'].astype(str) + df['unknown'].astype(str)

# Remove 'min' from 'id' string
df['id'] = df['id'].str.replace('min', '')

labels = df['id'].unique()

fig, axs = plt.subplots(1, 2, figsize=(15, 6))

for label in labels:
    subset = df[df['id'] == label]
    if label.find("jobs_done=2") != -1:
        ax = axs[0] # select first subplot
        if label.find("(fifth)") != -1:
            label = "jobs=2 (fifth)"
            ax.plot(subset['percentage'], subset['safety_measure'],  '-x', label=label)
        elif label.find("(fourth)") != -1:
            label = "jobs=2 (fourth)"
            ax.plot(subset['percentage'], subset['safety_measure'],  '-|', label=label)
        elif label.find("(third)") != -1:
            label = "jobs=2 (third)"
            ax.plot(subset['percentage'], subset['safety_measure'],  '-s', label=label)
        elif label.find("(second)") != -1:
            label = "jobs=2 (second)"
            ax.plot(subset['percentage'], subset['safety_measure'],  '-*', label=label)
        else:
            label = "jobs=2 (first)"
            ax.plot(subset['percentage'], subset['safety_measure'],  '-o', label=label)
        ax.set_xticks(np.arange(0, 1+0.05, 0.1))
        ax.set_yticks(np.arange(0, 1+0.05, 0.1))
        ax.grid(True, alpha=0.2)
        ax.set_xlabel('Percentage')
        ax.set_ylabel('Safety Measure')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    elif label.find("empty") != -1:
        ax = axs[1] # select second subplot
        if label.find("(fifth)") != -1:
            label = "empty (fifth)"
            ax.plot(subset['percentage'], subset['safety_measure'],  '-x', label=label)
        elif label.find("(fourth)") != -1:
            label = "empty (fourth)"
            ax.plot(subset['percentage'], subset['safety_measure'],  '-|', label=label)
        elif label.find("(third)") != -1:
            label = "empty (third)"
            ax.plot(subset['percentage'], subset['safety_measure'],  '-s', label=label)
        elif label.find("(second)") != -1:
            label = "empty (second)"
            ax.plot(subset['percentage'], subset['safety_measure'],  '-*', label=label)
        else:
            label = "empty (first)"
            ax.plot(subset['percentage'], subset['safety_measure'],  '-o', label=label)
        ax.set_xticks(np.arange(0, 1+0.05, 0.1))
        ax.set_yticks(np.arange(0, 1+0.05, 0.1))
        ax.grid(True, alpha=0.2)
        ax.set_xlabel('Percentage')
        ax.set_ylabel('Safety Measure')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')


fig.subplots_adjust(wspace=0.5)
# Save the figure to png
plt.savefig('plot_combined.png', dpi=300, bbox_inches='tight')
