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

# Calculate global IQR and threshold for Kalium (K)
K_data = df['K']
q1 = K_data.quantile(0.25)
q3 = K_data.quantile(0.75)
iqr = q3 - q1
upper_limit = q3 + 1.5 * iqr

# Create data after removing outliers based on global threshold
df_after = df[df['K'] <= upper_limit]

# Set up the side-by-side subplots
fig, axes = plt.subplots(1, 2, figsize=(18, 8), sharey=True)
sns.set_theme(style="whitegrid")

# Sort crops alphabetically for consistency
crop_order = sorted(df['label'].unique())

# Subplot 1: Sebelum IQR Removal
sns.boxplot(data=df, x='K', y='label', ax=axes[0], order=crop_order, palette='viridis')
axes[0].axvline(upper_limit, color='red', linestyle='--', linewidth=2, 
                label=f'Global Outlier Threshold ({upper_limit:.1f} mg/kg)')
axes[0].set_title('A. Sebelum IQR Removal (Outlier Dipertahankan)', fontsize=14, weight='bold', pad=10)
axes[0].set_xlabel('Kadar Kalium (K) (mg/kg)', fontsize=12)
axes[0].set_ylabel('Komoditas (Crop Label)', fontsize=12)
axes[0].legend(loc='lower right', fontsize=10)

# Subplot 2: Setelah IQR Removal
# We still plot all classes in crop_order to show that Apple & Grapes are missing/deleted
sns.boxplot(data=df_after, x='K', y='label', ax=axes[1], order=crop_order, palette='viridis')
axes[1].axvline(upper_limit, color='red', linestyle='--', linewidth=2, 
                label=f'Global Outlier Threshold ({upper_limit:.1f} mg/kg)')
axes[1].set_title('B. Setelah IQR Removal (Semua K > 92.5 mg/kg Dihapus)', fontsize=14, weight='bold', pad=10)
axes[1].set_xlabel('Kadar Kalium (K) (mg/kg)', fontsize=12)
axes[1].set_ylabel('') # Remove y-label for the second subplot as it shares y-axis
axes[1].legend(loc='lower right', fontsize=10)

plt.suptitle('Dampak Pembersihan Outlier Global (IQR) terhadap Data Agronomi Kalium (K)', 
             fontsize=16, weight='bold', y=0.98)
plt.tight_layout()

# Find parent path (root folder) to save the image
output_dir = os.path.dirname(csv_path) if os.path.dirname(csv_path) else '.'
output_path = os.path.join(output_dir, 'outlier_comparison.png')

plt.savefig(output_path, dpi=300)
plt.close()

print(f"Outlier comparison boxplot successfully saved to {output_path}.")
