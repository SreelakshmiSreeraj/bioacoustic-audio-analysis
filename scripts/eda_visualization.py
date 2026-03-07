import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("../features/features_all.csv")

print(df.head())
print(df.shape)

# Plot 1 — Category distribution
plt.figure(figsize=(6,4))
sns.countplot(x="category", data=df)

plt.title("Distribution of Sound Categories")
plt.xlabel("Category")
plt.ylabel("Count")

plt.show()

# Plot 2 — Spectral centroid distribution

plt.figure(figsize=(6,4))
sns.histplot(data=df, x="spectral_centroid", hue="category", bins=50)
plt.title("Spectral Centroid Distribution")
plt.show()

#Plot 3 — MFCC feature comparison
plt.figure(figsize=(6,4))
sns.boxplot(x="category", y="mfcc0", data=df)
plt.title("MFCC0 Distribution by Category")
plt.show()

#Plot 4 — Feature correlation heatmap
plt.figure(figsize=(12,8))

numeric_df = df.select_dtypes(include=["float64","int64"])

sns.heatmap(numeric_df.corr(), cmap="coolwarm")

plt.title("Feature Correlation Matrix")
plt.show()

#Plot 5 — Bioacoustic activity comparison
plt.figure(figsize=(6,4))
sns.boxplot(x="category", y="aci", data=df)

plt.title("Acoustic Complexity Index by Category")
plt.show()