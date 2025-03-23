import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
import umap
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import umap.umap_ as umap  # Correct way to import UMAP


import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# Load your dataset
df = pd.read_csv("ASD_relative_abundance_with_labels.csv")  # Replace with your actual file

# Drop non-numeric columns
df_numeric = df.select_dtypes(include=[np.number])  # Keep only numeric columns

# Standardize the numeric features
scaler = StandardScaler()
features_scaled = scaler.fit_transform(df_numeric)

# Print the transformed features
print(features_scaled)

# Apply t-SNE
tsne = TSNE(n_components=2, perplexity=30, random_state=42)
tsne_result = tsne.fit_transform(features_scaled)

# Apply UMAP
umap_model = umap.UMAP(n_components=2, random_state=42)
umap_result = umap_model.fit_transform(features_scaled)

# K-Means Clustering
kmeans = KMeans(n_clusters=2, random_state=42)
clusters = kmeans.fit_predict(features_scaled)

# Convert results to DataFrame
df["TSNE1"], df["TSNE2"] = tsne_result[:, 0], tsne_result[:, 1]
df["UMAP1"], df["UMAP2"] = umap_result[:, 0], umap_result[:, 1]
df["Cluster"] = clusters  # K-Means clusters

# Plot t-SNE
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
sns.scatterplot(x="TSNE1", y="TSNE2", hue=df["Cluster"], style=df["ASD"], palette="viridis", data=df)
plt.title("t-SNE Visualization of Clusters")

# Plot UMAP
plt.subplot(1, 2, 2)
sns.scatterplot(x="UMAP1", y="UMAP2", hue=df["Cluster"], style=df["ASD"], palette="coolwarm", data=df)
plt.title("UMAP Visualization of Clusters")

plt.show()
