import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Data for Stage 2 multilabel metrics
metrics = ['Subset Accuracy\n(Exact Match)', 'F1-Score\n(Micro)', 'F1-Score\n(Macro)']
kmeans_vals = [0.8402, 0.9639, 0.9566]
gmm_vals = [0.8970, 0.9662, 0.9657]

x = np.arange(len(metrics))
width = 0.35

# Plot setup
fig, ax = plt.subplots(figsize=(10, 6))
sns.set_theme(style="whitegrid")

# Create bars
rects1 = ax.bar(x - width/2, kmeans_vals, width, label='K-Means (12 Clusters)', color='#80b1d3')
rects2 = ax.bar(x + width/2, gmm_vals, width, label='GMM (12 Clusters)', color='#bc80bd')

# Add titles and labels
ax.set_title('Performance Comparison of Stage 2 Multilabel Recommendation Models', fontsize=14, weight='bold', pad=15)
ax.set_ylabel('Score (Percentage / Value)', fontsize=12)
ax.set_xticks(x)
ax.set_xticklabels(metrics, fontsize=11)
ax.set_ylim(0, 1.1)  # Give space for annotations

# Add legend
ax.legend(loc='lower left', frameon=True, facecolor='white', edgecolor='none', fontsize=11)

# Function to add labels on top of the bars
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height*100:.2f}%',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', weight='bold', fontsize=10)

autolabel(rects1)
autolabel(rects2)

plt.tight_layout()

# Find parent path (root folder) to save the image
output_path = 'model_comparison_bar.png'
if os.path.exists('..'):
    output_path = os.path.join('..', 'model_comparison_bar.png')

plt.savefig(output_path, dpi=300)
print(f"Bar chart successfully saved to {output_path}.")
