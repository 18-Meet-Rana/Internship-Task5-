# Import libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load dataset
df = pd.read_csv("train.csv")  # Make sure train.csv is in the same folder
print(df.head())

# Basic info
print("\n--- Dataset Info ---")
print(df.info())

print("\n--- Summary Statistics ---")
print(df.describe())

# Check missing values
print("\n--- Missing Values ---")
print(df.isnull().sum())

# Univariate Analysis - Age Distribution
plt.figure(figsize=(6,4))
df['Age'].hist(bins=30, color='skyblue', edgecolor='black')
plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Count")
plt.show()

# Boxplot: Age vs Survived
plt.figure(figsize=(6,4))
sns.boxplot(x='Survived', y='Age', data=df)
plt.title("Age vs Survival")
plt.show()

# Countplot: Survival by Gender
plt.figure(figsize=(6,4))
sns.countplot(x='Survived', hue='Sex', data=df, palette='Set2')
plt.title("Survival Count by Gender")
plt.show()

# Correlation Heatmap
plt.figure(figsize=(8,6))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

# Pairplot
sns.pairplot(df[['Age', 'Fare', 'Survived']], hue='Survived', palette='husl')
plt.show()

# Skewness check
from scipy.stats import skew
print("\nSkewness of Fare:", skew(df['Fare'].dropna()))

# Log transform if skewed
df['Fare_log'] = np.log1p(df['Fare'])
