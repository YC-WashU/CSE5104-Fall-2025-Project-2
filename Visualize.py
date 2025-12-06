import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load Data
df = pd.read_csv('winequality-white.csv', sep=';')

# 2. Visualize Distributions
plt.figure(figsize=(15, 12))
df.hist(bins=30, figsize=(15, 12), layout=(4, 3), edgecolor='black')
plt.tight_layout()
plt.savefig('Data Visualization.png') 
