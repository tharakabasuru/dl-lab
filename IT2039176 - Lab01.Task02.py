#Task-02


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv('Au_nanoparticle_dataset.csv')

#01
new_df = data[['N_total', 'N_bulk', 'N_surface', 'R_avg']]

#02
print(new_df.head(20))

#03
mean_values = new_df.mean()
std_values = new_df.std()
quartiles = new_df.quantile([0.25, 0.5, 0.75])

print("Mean values:\n", mean_values)
print("\nStandard deviation values:\n", std_values)
print("\nQuartile values:\n", quartiles)

#04
plt.figure(figsize=(20, 5))
for i, column in enumerate(new_df.columns):
    plt.subplot(1, 4, i + 1)
    plt.hist(new_df[column], bins=30, alpha=0.7, color='blue')
    plt.title(column)
    plt.xlabel('Value')
    plt.ylabel('Frequency')
plt.tight_layout()
plt.show()

#05
sns.pairplot(new_df)
plt.show()

#06
g = sns.PairGrid(new_df)
g.map_upper(sns.histplot) 
g.map_diag(sns.histplot, kde=True) 
g.map_lower(sns.kdeplot) 
plt.show()

