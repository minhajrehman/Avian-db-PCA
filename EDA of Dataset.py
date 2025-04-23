
# Exploratory Data Analysis (EDA) of  Dataset 

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
sns.set_theme(style="whitegrid")

# Load the updated dataset
df = pd.read_excel('Dataset.xlsx', engine='openpyxl')

# Data Cleaning
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
df.columns = df.columns.str.strip().str.replace('%', 'Percent').str.replace('.', '', regex=False).str.replace(' ', '_')
df.dropna(inplace=True)

# --- Univariate Analysis ---

# 1. Total Death Distribution
plt.figure(figsize=(8, 5))
sns.histplot(df['Total_Death'], kde=True, color='darkred', bins=30)
plt.title('Distribution of Total Deaths', fontsize=14, weight='bold')
plt.xlabel('Total Deaths')
plt.ylabel('Frequency')
plt.tight_layout()
plt.show()

# 2. Region Distribution
plt.figure(figsize=(10, 5))
sns.countplot(x='Region', hue='Region', data=df, palette='Set3', edgecolor="black", legend=False)
plt.xticks(rotation=45, ha='right')
plt.title('Frequency of Observations by Region')
plt.tight_layout()
plt.show()

# --- Bivariate Analysis ---

# 3. Correlation Heatmap (includes Total_Death)
plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap Including Total Death')
plt.tight_layout()
plt.show()

# 4. Total Death vs Avg_Temp
plt.figure(figsize=(8, 6))
sns.scatterplot(x='Avg_Temp', y='Total_Death', hue='Region', data=df)
plt.title('Total Death vs Average Temperature')
plt.tight_layout()
plt.show()

# 5. Total Death vs Humidity
plt.figure(figsize=(8, 6))
sns.scatterplot(x='HumidityPercent', y='Total_Death', hue='Region', data=df)
plt.title('Total Death vs Humidity')
plt.tight_layout()
plt.show()

# 6. Average Total Death by Region
avg_death_by_region = df.groupby('Region')['Total_Death'].mean().sort_values()
plt.figure(figsize=(10, 6))
avg_death_by_region.plot(kind='barh', color='crimson')
plt.title('Average Total Deaths by Region')
plt.xlabel('Total Deaths')
plt.tight_layout()
plt.show()

# 7. Total Deaths by Year 
if 'Year' in df.columns:
    yearly_death = df.groupby('Year')['Total_Death'].sum().sort_index()
    plt.figure(figsize=(10, 6))
    yearly_death.plot(kind='bar', color='teal', edgecolor='black')
    plt.title('Total Deaths by Year')
    plt.xlabel('Year')
    plt.ylabel('Total Deaths')
    plt.tight_layout()
    plt.show()

