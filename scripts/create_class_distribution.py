import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Helper function to find files in current or parent directory
def find_file(filename):
    if os.path.exists(filename):
        return filename
    parent_path = os.path.join('..', filename)
    if os.path.exists(parent_path):
        return parent_path
    return filename

# Load dataset
csv_path = find_file('crop_recommendation.csv')
df = pd.read_csv(csv_path)

# Count samples per crop label
class_counts = df['label'].value_counts().sort_index()

# Plot horizontal bar chart
plt.figure(figsize=(12, 10))
sns.set_theme(style="whitegrid")

# Create barplot with a nice color palette
colors = sns.color_palette("viridis", len(class_counts))
bars = plt.barh(class_counts.index[::-1], class_counts.values[::-1], color=colors, edgecolor='black', height=0.7)

# Add text labels on the bars
for bar in bars:
    width = bar.get_width()
    plt.text(width - 8, bar.get_y() + bar.get_height()/2, f'{int(width)} samples', 
             va='center', ha='right', color='white', weight='bold', fontsize=10)

plt.title('Sample Count Distribution of the 22 Crop Classes (Perfectly Balanced)', fontsize=14, weight='bold', pad=15)
plt.xlabel('Number of Samples (Frequency)', fontsize=12)
plt.ylabel('Crop Type (Label)', fontsize=12)
plt.xlim(0, 115) # Give some space on the right for clean layout

plt.tight_layout()

# Find parent path (root folder) to save the image
output_dir = os.path.dirname(csv_path) if os.path.dirname(csv_path) else '.'
output_path = os.path.join(output_dir, 'crop_class_distribution.png')

plt.savefig(output_path, dpi=300)
plt.close()

print(f"Crop class distribution plot successfully saved to {output_path}.")
