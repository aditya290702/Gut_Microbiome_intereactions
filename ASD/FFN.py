import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# ----------------------------
# 1. Load and Preprocess Data
# ----------------------------
# Load your cleaned microbial abundance CSV (samples as rows, species as columns)
df = pd.read_csv("ASD_cleaned_dataset.csv", sep=",", index_col=0)

# Assume data is in abundance counts or relative abundances
# Apply log transformation (log1p to avoid log(0)) and then standard scaling.
data = np.log1p(df.values)  # shape: (n_samples, n_features)
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

# Convert to PyTorch tensor
X = torch.tensor(data_scaled, dtype=torch.float)
n_samples, n_features = X.shape
print(f"Loaded data with {n_samples} samples and {n_features} species.")

# ----------------------------
# 2. Build the Autoencoder using FFN
# ----------------------------
# Define the encoder and decoder using Sequential modules.
encoder = nn.Sequential(
    nn.Linear(n_features, 64),
    nn.ReLU(),
    nn.Linear(64, 16),  # Latent dimension of 16 (adjust as needed)
    nn.ReLU()
)
decoder = nn.Sequential(
    nn.Linear(16, 64),
    nn.ReLU(),
    nn.Linear(64, n_features)
)


# Combine them into an autoencoder function
def autoencoder(x):
    z = encoder(x)
    x_reconstructed = decoder(z)
    return z, x_reconstructed


# ----------------------------
# 3. Train the Autoencoder
# ----------------------------
criterion = nn.MSELoss()  # Reconstruction loss
optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=0.01)
num_epochs = 100

for epoch in range(1, num_epochs + 1):
    encoder.train()
    decoder.train()
    optimizer.zero_grad()
    latent, output = autoencoder(X)
    loss = criterion(output, X)
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print(f"Epoch {epoch}/{num_epochs}, Loss: {loss.item():.4f}")

# ----------------------------
# 4. Extract Latent Representations & Cluster
# ----------------------------
# Set model in evaluation mode and get latent features
encoder.eval()
with torch.no_grad():
    latent_features = encoder(X).numpy()

# Use K-Means to cluster the latent space representations
n_clusters = 3  # Adjust the number of clusters as needed
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
clusters = kmeans.fit_predict(latent_features)
print("Cluster assignments for each sample:")
print(clusters)

# ----------------------------
# 5. Visualize the Latent Space
# ----------------------------
# For visualization, we can reduce to 2D using PCA (or t-SNE)
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
latent_2d = pca.fit_transform(latent_features)

plt.figure(figsize=(8, 6))
scatter = plt.scatter(latent_2d[:, 0], latent_2d[:, 1], c=clusters, cmap='viridis', s=50)
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("2D Visualization of Latent Space with K-Means Clusters")
plt.legend(*scatter.legend_elements(), title="Clusters")
plt.show()

# ----------------------------
# 6. Identifying Keystone Species (Feature Importance)
# ----------------------------
# One approach is to examine the weights of the first encoder layer.
# Higher absolute weights for a given species might indicate higher influence in the latent representation.
first_layer_weights = encoder[0].weight.detach().numpy()  # shape: (64, n_features)
# Compute average absolute weight per species (column)
avg_abs_weights = np.mean(np.abs(first_layer_weights), axis=0)
feature_importance = pd.Series(avg_abs_weights, index=df.columns)
keystone_species = feature_importance.sort_values(ascending=False).head(10)
print("Top 10 keystone species (by average absolute weight):")
print(keystone_species)
