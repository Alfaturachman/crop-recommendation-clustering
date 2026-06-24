import os
import pandas as pd
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

# Numeric columns representing the 7 features
numeric_cols = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']

# Set style
sns.set_theme(style="white")

plt.figure(figsize=(10, 8))
corr_matrix = df[numeric_cols].corr()

# Draw heatmap
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5, 
            square=True, cbar_kws={"shrink": .8})

plt.title("Matriks Korelasi Fitur Agronomi", fontsize=14, weight='bold', pad=15)
plt.tight_layout()

# Find parent path (root folder) to save the image
output_dir = os.path.dirname(csv_path) if os.path.dirname(csv_path) else '.'
output_path = os.path.join(output_dir, 'feature_correlation.png')

plt.savefig(output_path, dpi=300)
plt.close()

print(f"Feature correlation matrix heatmap successfully saved to {output_path}.")
