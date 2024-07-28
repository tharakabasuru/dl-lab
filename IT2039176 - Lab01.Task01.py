#Task-01


#1.
import numpy as np
array_4x4 = np.random.exponential(scale=1.0, size=(4, 4))
print("4x4 array from exponential distribution:\n", array_4x4)


#2.
import matplotlib.pyplot as plt

#random arrays
exp_data = np.random.exponential(scale=1.0, size=100000)
uniform_data = np.random.uniform(low=-3.0, high=3.0, size=100000)
normal_data = np.random.normal(loc=0.0, scale=1.0, size=100000)

#histograms
plt.figure(figsize=(10, 6))

#Exponential distribution histogram
plt.hist(exp_data, density=True, bins=100, histtype="step", color="blue", label="Exponential")

#Uniform distribution histogram
plt.hist(uniform_data, density=True, bins=100, histtype="step", color="green", label="Uniform")

#Normal distribution histogram
plt.hist(normal_data, density=True, bins=100, histtype="step", color="red", label="Normal")

plt.axis([-3.5, 3.5, 0, 1.1])

plt.legend(loc="upper right")
plt.title("Random Distributions")
plt.xlabel("Value")
plt.ylabel("Density")

plt.show()


#3.
from mpl_toolkits.mplot3d import Axes3D

#Create a meshgrid
x = np.linspace(-5, 5, 100)
y = np.linspace(-5, 5, 100)
X, Y = np.meshgrid(x, y)
Z = X**2 + Y**2

#3D surface
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, cmap='viridis')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('$Z = X^2 + Y^2$')

plt.show()


#4.
import pandas as pd
import seaborn as sns

data = pd.read_csv('online_store_customer_data.csv')

features = ['Age', 'Referal', 'Amount_spent']
data_selected = data[features]

pearson_corr = data_selected.corr(method='pearson')

spearman_corr = data_selected.corr(method='spearman')

#Pearson correlation heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(pearson_corr, annot=True, fmt='.2f', cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Pearson Correlation Coefficient')
plt.show()

#Spearman correlation heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(spearman_corr, annot=True, fmt='.2f', cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Spearman Rank Correlation')

plt.show()
