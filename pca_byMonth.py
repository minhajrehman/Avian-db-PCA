import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Load the dataset
data = pd.read_excel("weather.xlsx")

# Select numerical features for PCA
features = ['Avg Temp', 'Max Temp', 'Min Temp', 'Temp Def', 'Humidity']
x = data[features].values

# Standardize the features
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

# Apply PCA
pca = PCA(n_components=2)
pca_result = pca.fit_transform(x_scaled)

# Create a PCA result DataFrame
pca_df = pd.DataFrame(data=pca_result, columns=['principal component 1', 'principal component 2'])
pca_df['Month'] = data['Month']

# Explained variance
print("Explained variation per principal component:", pca.explained_variance_ratio_)

# Plot PCA result by month
plt.figure(figsize=(10,10))
plt.xticks(fontsize=12)
plt.yticks(fontsize=14)
plt.xlabel('Principal Component - 1', fontsize=16)
plt.ylabel('Principal Component - 2', fontsize=16)
plt.title('PCA of Weather Dataset by Month', fontsize=18)

# Assign colors using a color map for 12 months
months = sorted(pca_df['Month'].unique())
colors = plt.cm.get_cmap('tab20', 12)

for i, month in enumerate(months):
    month_data = pca_df[pca_df['Month'] == month]
    plt.scatter(month_data['principal component 1'],
                month_data['principal component 2'],
                s=50,
                label=f'Month {month}',
                color=colors(i))

plt.legend(loc='best', fontsize=12, title='Month')
plt.grid()
plt.show()
