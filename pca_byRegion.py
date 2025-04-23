import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Load the dataset
data = pd.read_excel("weather.xlsx")

# Select features for PCA (excluding categorical and temporal)
features = ['Avg Temp', 'Max Temp', 'Min Temp', 'Temp Def', 'Humidity']
x = data[features].values

# Standardize the features
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

# Perform PCA to reduce to 2 components
pca = PCA(n_components=2)
pca_result = pca.fit_transform(x_scaled)

# Create a DataFrame for PCA results
pca_df = pd.DataFrame(data=pca_result, columns=['principal component 1', 'principal component 2'])
pca_df['Region'] = data['Region']

# Explained variance
print("Explained variation per principal component:", pca.explained_variance_ratio_)

# Plotting
plt.figure(figsize=(10,10))
plt.xticks(fontsize=12)
plt.yticks(fontsize=14)
plt.xlabel('Principal Component - 1', fontsize=16)
plt.ylabel('Principal Component - 2', fontsize=16)
plt.title('PCA of Weather Dataset by Region', fontsize=18)

# Unique regions and colors for plotting
regions = pca_df['Region'].unique()
colors = plt.cm.get_cmap('tab10', len(regions))

for i, region in enumerate(regions):
    region_data = pca_df[pca_df['Region'] == region]
    plt.scatter(region_data['principal component 1'],
                region_data['principal component 2'],
                s=50,
                label=region,
                color=colors(i))

plt.legend(loc='best', fontsize=12)
plt.grid()
plt.show()
