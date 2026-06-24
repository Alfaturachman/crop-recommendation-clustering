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
features = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
feature_names = {
    'N': 'Nitrogen (N) - mg/kg',
    'P': 'Fosfor (P) - mg/kg',
    'K': 'Kalium (K) - mg/kg',
    'temperature': 'Suhu (Temperature) - °C',
    'humidity': 'Kelembaban (Humidity) - %',
    'ph': 'Tingkat Keasaman (pH)',
    'rainfall': 'Curah Hujan (Rainfall) - mm'
}

# Set style
sns.set_theme(style="whitegrid")
fig, axes = plt.subplots(4, 2, figsize=(15, 18))
axes = axes.ravel()

# Custom colors for each feature to make it look premium
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']

for i, col in enumerate(features):
    ax = axes[i]
    # Plot histogram with KDE
    sns.histplot(data=df, x=col, kde=True, ax=ax, color=colors[i], bins=30, edgecolor='black', alpha=0.7)
    
    # Customize labels and titles
    ax.set_title(f'Distribusi {feature_names[col]}', fontsize=12, weight='bold', pad=10)
    ax.set_xlabel('')
    ax.set_ylabel('Frekuensi (Jumlah Sampel)', fontsize=10)

# Delete the 8th subplot since we only have 7 features
fig.delaxes(axes[7])

plt.suptitle('Sebaran Distribusi Frekuensi 7 Variabel Agronomi & Mikroklimat', fontsize=16, weight='bold', y=0.98)
plt.tight_layout()

# Find parent path (root folder) to save the image
output_dir = os.path.dirname(csv_path) if os.path.dirname(csv_path) else '.'
output_path = os.path.join(output_dir, 'feature_distributions.png')

plt.savefig(output_path, dpi=300)
plt.close()

print(f"Feature distributions plot successfully saved to {output_path}.")
